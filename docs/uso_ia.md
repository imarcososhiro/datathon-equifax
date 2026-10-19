# Uso de IA

O edital pede transparência sobre o uso de IA. Usamos o Claude (Anthropic) ao longo do projeto como apoio para organizar o código, discutir resultados e revisar textos. As decisões são da equipe, e tudo o que a IA sugeriu ou escreveu foi revisado por um de nós antes de entrar no projeto.

**Dados:** seguindo a regra confirmada com a organização, a IA nunca recebeu linhas das bases, `id` ou `cep8`. Ela trabalhou só com o código, os nomes das colunas e as saídas agregadas dos notebooks (métricas, contagens, tabelas e gráficos).

## Organização do repositório

A IA nos ajudou a montar a estrutura geral do repositório: o pacote `src/equifax/`, com os módulos de funções separados por etapa, e pastas próprias para os notebooks, as figuras e tabelas geradas (`reports/`), a documentação (`docs/`) e os testes, além dos arquivos de configuração (`config.py`, `requirements.txt`, `.gitignore`). Assim cada coisa tem um lugar fixo, e os notebooks leem e gravam sempre nos mesmos caminhos.

## Organização dos notebooks

Usamos a IA para tirar dos notebooks o código repetido: as tabelas e os gráficos viraram funções reutilizáveis, chamadas da mesma forma em todos eles. A IA também sugeriu gráficos com mais elementos para melhorar a leitura (por exemplo, o número de clientes em cada célula dos mapas de calor).

## Em cada notebook

- **Todos:** a IA revisou e complementou os textos de conclusão que escrevemos e sugeriu os markdowns que faltavam para o notebook se ler em sequência.
- **01 (EDA):** a IA serviu de segunda opinião na leitura dos gráficos e das tabelas e nos ajudou a aprofundar algumas análises. A descoberta dos códigos especiais nas VPs (`-1`, `-2` e `9999`) veio da análise da IA.
- **Desvio entre 3 sementes:** a ideia de repetir cada comparação com 3 sementes e usar o desvio entre elas como medida de ruído foi da IA. Foi uma boa ideia: aprofundou os resultados e guiou as nossas decisões do 02 em diante, porque uma mudança só conta como ganho quando supera esse desvio (no corte de variáveis, nas features do M2, no M3 e na escolha do vencedor).
- **02 (seleção de variáveis):** a IA ajudou principalmente na decisão sobre as variáveis com IV abaixo de 0,02. Em vez de cortar todas, sai só a que também tem ganho no modelo abaixo da mediana do ganho das demais. A regra foi sugerida pela IA; sem ela, teríamos feito um corte bem mais simplista. A IA também ajudou na análise dos códigos especiais e na interpretação das variáveis instáveis, safra a safra. Isso deu mais profundidade às análises, sempre com o controle e a decisão final com a equipe.
- **03 (modelo base):** a IA ajudou principalmente na visualização dos resultados dos modelos e na construção do notebook, para que ele rodasse o menor número de vezes possível, já que as buscas de hiperparâmetros levam cerca de 2 horas.
- **04 (feature engineering):** a IA propôs as interações renda × idade e idade × região e as variáveis de renda relativa ao estado e à região. Também nos ajudou a decidir o que usar no M2.
- **05 (interpretabilidade e escoragem):** a IA ajudou a melhorar a visualização dos resultados e na interpretação completa deles, resumida no fim do notebook.
- **06 (modelo macro):** a IA ajudou na visualização dos resultados e na tomada de decisão a partir da interpretação deles.
- **07 (modelo final):** a IA ajudou na visualização dos resultados (gráficos e tabelas) e no resumo que fecha o notebook.

## Relatório e documentação

A IA ajudou a montar o relatório técnico e este documento de uso de IA, a partir dos nossos documentos e das saídas dos notebooks. O conteúdo e as conclusões foram revisados pela equipe.

## Versão pública do repositório

A IA ajudou a montar a versão pública deste repositório, sem os resultados da base, e o script `scripts/visuais_publicos.py`, que gera as imagens do README a partir do método, das séries macro públicas e de uma base sintética.

## Apresentação

Usamos a IA para fazer a base da apresentação em PowerPoint (`.pptx`), a partir do relatório e dos resultados dos notebooks. Em cima dessa base, revisamos e ajustamos os slides.
