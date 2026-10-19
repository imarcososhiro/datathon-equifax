# Catálogo de séries macroeconômicas (Passo 7, bloco D)

`[IA]` Redigido com auxílio de IA (Claude) em 11/10/2026. Códigos conferidos nos portais do BCB e do IBGE; os valores serão conferidos de novo na primeira coleta (`resumo` em `src/equifax/macro.py`).

O catálogo usado pelo código fica em `src/equifax/macro.py` (`CATALOGO`). Este documento explica cada série e a regra de disponibilidade. Com só 10 safras, entram poucas séries, cada uma com uma hipótese clara.

## Regra de disponibilidade (sem vazamento)

- A safra só tem o mês, então cada concessão é tratada como feita no **dia 1 do mês da safra**.
- Uma observação do mês de referência *r* só pode ser usada a partir do dia 1 do mês *r + defasagem*. A defasagem é fixa por série e conservadora: ela cobre a data normal de divulgação com folga, porque não temos o calendário histórico de cada divulgação.
- Séries diárias (Selic) valem a partir do dia seguinte ao da observação. Para a safra *m*, vale o último valor do mês anterior.
- Nas séries trimestrais, o mês de referência é o último mês do trimestre (o 2º trimestre de 2023 é junho de 2023).
- A coleta guarda a série inteira com a data do download. Usamos a versão atual de cada série, e não a versão que existia na data da concessão. Nas séries revisadas (crédito do BCB), isso é uma limitação conhecida; a revisão costuma ser pequena.

## Séries

| Série (`nome`) | Fonte e código | Unidade | Frequência | Geografia | Divulgação normal | Defasagem usada | Revisada? | Hipótese |
|---|---|---|---|---|---|---|---|---|
| `selic_meta` | BCB SGS 432 | % a.a. | Diária | Brasil | Na noite da reunião do Copom | Dia seguinte | Não | Juro alto encarece o crédito e aperta o orçamento |
| `ipca_mes` | BCB SGS 433 (IBGE) | % no mês | Mensal | Brasil | Cerca do dia 10 do mês seguinte | 2 meses | Não | Inflação reduz a renda real |
| `ipca_12m` | BCB SGS 13522 (IBGE) | % em 12 meses | Mensal | Brasil | Cerca do dia 10 do mês seguinte | 2 meses | Não | Idem, sem o ruído mensal |
| `inadimplencia_pf` | [BCB SGS 21084](https://dadosabertos.bcb.gov.br/dataset/21084-inadimplencia-da-carteira-de-credito---pessoas-fisicas---total) | % | Mensal | Brasil | Fim do mês seguinte (Estatísticas Monetárias e de Crédito) | 3 meses | Sim | Retrato direto do ciclo de crédito |
| `comprometimento_renda` | [BCB SGS 29034](https://dadosabertos.bcb.gov.br/dataset/29034-comprometimento-de-renda-das-familias-com-o-servico-da-divida-com-o-sistema-financeiro-nacion) | % | Mensal | Brasil | Até 8 semanas após o mês | 4 meses | Sim | Famílias com mais renda comprometida pagam pior |
| `endividamento_familias` | [BCB SGS 29037](https://dadosabertos.bcb.gov.br/dataset/29037-endividamento-das-familias-com-o-sistema-financeiro-nacional-em-relacao-a-renda-acumulada-dos) | % | Mensal | Brasil | Até 8 semanas após o mês | 4 meses | Sim | Famílias já endividadas pagam pior |
| `desocupacao_br` | IBGE SIDRA, tabela 6381, variável 4099 (PNAD Contínua) | % | Trimestre móvel | Brasil | Fim do mês seguinte ao último mês do trimestre | 3 meses | Raramente | Desemprego derruba a capacidade de pagar |
| `desocupacao_uf` | IBGE SIDRA, tabela 4099, variável 4099 (PNAD Contínua) | % | Trimestral | UF | Meados ou fim do 2º mês após o trimestre | 4 meses | Raramente | Idem, e varia entre clientes da mesma safra |

URLs usadas: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.{código}/dados?formato=json&dataInicial=...&dataFinal=...` e `https://apisidra.ibge.gov.br/values/t/{tabela}/{n1 ou n3}/all/v/4099/p/all`. A janela de coleta vai de 01/2020 a 12/2024: cobre a variação de 12 meses, a maior defasagem antes de 202301 e as safras da OOT cega.

## Exemplo da regra

Para a safra 202303 (concessão tratada como feita em 01/03/2023):

| Série | Último mês de referência que pode ser usado |
|---|---|
| `selic_meta` | Valor em 28/02/2023 |
| `ipca_mes`, `ipca_12m` | 01/2023 |
| `inadimplencia_pf`, `desocupacao_br` | 12/2022 |
| `comprometimento_renda`, `endividamento_familias` | 11/2022 |
| `desocupacao_uf` | 3º trimestre de 2022 (referência 09/2022, que vale a partir de 01/01/2023) |

## Junção com a base (`montar_tabela_macro` e `juntar_macro`)

1. `montar_tabela_macro()` lê o download mais recente e monta uma tabela longa: série, mês (ou dia) de referência, UF (`BR` nas nacionais), valor e `disponivel_em`, o primeiro dia em que o valor pode ser usado (referência + defasagem; Selic, dia seguinte).
2. Acrescenta as transformações, cada uma com a sua própria data de disponibilidade: variação em 3 meses (IPCA 12 meses, inadimplência PF), variação em 12 meses (inadimplência PF, comprometimento, endividamento, desocupação Brasil e UF) e a diferença da desocupação da UF para a do Brasil no mesmo trimestre (vale quando as duas valem).
3. `juntar_macro(base, tabela)` dá a cada linha, para cada série, o último valor com `disponivel_em <=` dia 1 do mês da safra. As séries por UF usam o `estado` do cliente; "Sem informação" recebe a série nacional (a diferença para o Brasil vira 0). Cada série original ganha também `<serie>_idade_meses`, os meses entre a referência usada e a concessão.
4. O resultado tem o mesmo índice e a mesma ordem da base, sem linhas a mais ou a menos. Um estado sem valor na série para com erro, e uma série sem dado disponível na data gera um aviso.

São 24 colunas (16 séries e 8 idades), todas com o prefixo `macro_`. Conferido com os dados reais: na safra 202303 entram exatamente os meses de referência da tabela de exemplo acima, e o primeiro corte da Selic (13,25%, em vigor desde 03/08/2023) só aparece a partir da safra 202309.

## Variação entre clientes e risco de "decorar o mês"

Só a `desocupacao_uf` varia entre clientes da mesma safra. As demais têm um único valor por safra: 8 valores no treino. Elas podem servir para o modelo reconhecer o mês, e não para medir risco. Por isso o M3 é avaliado em safras posteriores às do treino (proposta: treino 202301–202306, avaliação 202307–202308), e não na divisão aleatória por cliente.

Linhas com `estado` igual a "Sem informação" recebem o valor nacional (`desocupacao_br` no lugar da UF).

## Fora do catálogo por enquanto

- **Inadimplência de pessoas físicas por UF:** o SGS só publica por estado a inadimplência de MEI e de empresas, não de pessoas físicas em geral. A alternativa é o SCR.data (planilhas mensais por UF), que dá mais trabalho; fica para depois, se o macro mostrar ganho.
- **IPCA por região metropolitana:** cobre só 16 áreas e deixaria várias UFs sem valor próprio.

## Em produção (texto-base para o relatório)

- A coleta roda agendada (por exemplo, toda semana) e grava uma nova pasta datada em `data/external/macro_bruto/`, com o manifesto.
- O modelo usa sempre o último valor disponível na data da concessão. Se uma divulgação atrasar 30 ou 60 dias, o valor anterior continua valendo, e a idade do dado (meses entre a referência e a concessão) é registrada.
- Um alerta dispara quando a idade de uma série passa da defasagem esperada mais um mês.
- Plano B: se uma fonte ficar indisponível por muito tempo, o score cai para o modelo sem macro (M2), que é mantido treinado em paralelo.
