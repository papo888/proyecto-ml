# Clasificación de Géneros Músicales usando Aprendizaje de Máquina basado en Caracteristicas Acústicas
## Gabriela Bejarano, Juan Pablo Parrado, Cristian Zea

## Descripción del Proyecto

El objetivo principal del proyecto es desarrollar un modelo de aprendizaje que clasificar canciones en diferentes géneros músicales usando sus caracteristicas acusticas. Se van a analizar variables que incluyen la energia, ritmo, acústica y valencia para identificar en que se diferencian los géneros músicales.

Para este proyecto se utilizaron varios modelos clasificación supervisada, luego se evaluaron para determinar que modelo es el que tiene un mejor desempeño. Tambien se analizan que variables son las que mas aportan valor a la clasificación de los generos.

## Descripción de la Problemática

Durante los años 90, se comenzó la digitalización de los productos musicales hacia el sector digital y, con los años ha crecido cada vez más la tendencia de los usuarios a escuchar su música por medio de plataformas digitales (Fink, 2021). Alrededor de tres de cada cuatro personas dijeron que escuchan música por medio de estas plataformas como, Spotify o Apple Music (El Tiempo, 2021).

Estas plataformas actualmente buscan una mejor experiencia para los usuarios, ayudando a que sus preferencias músicales sean fáciles de encontrar para poder retener más usuarios. Esta búsqueda de organización y recomendación de contenido suele depender de etiquetas manuales o criterios subjetivos, lo que puede ocasionar errores o inconsistencias.

Este proyecto busca solucionar esta problemática usando técnicas de machine learning, permitiendo una clasificación basada en los datos.

## Metricas

Este problema corresponde a una clasificaciónn de diferentes géneros, lo que indica un alto desbalanceo entre los diferentes generos, ya que existen generos con muchas canciones y algunos generos que con tienen una menor cantidad. Teniendo esto encuenta elegimos como métrica principal el Recall ya que mide de todas las canciones cuales realmente son de un genero, que tantas fue capaz de encontrar el modelo. En el caso de que el modelo llegue a ignorar un género con pocas canciones el Recall será muy bajo y al priorizarlo, ayudamos a que el modelo no deje por fuera ninguna canción de esos géneros dificies de clasificar.

Adicionalmente, como medida secundaria utilizaremos el F1-score, para asegurarnos que el modelo sea equilibrado y que adicionalmente sea exacto cuando predice algo.

## Referencias

Entrada de Blog (WIPO):

Fink, C. (2021, 23 de noviembre). El muestreo musical en la era digital. OMPI: La Economía de la Propiedad Intelectual. https://www.wipo.int/es/web/economics/w/blogs/music-sampling-in-the-digital-age

Artículo de Periódico Online (El Tiempo):

El Tiempo. (2021, 30 de noviembre). En datos: así escuchan música las personas en el mundo. https://www.eltiempo.com/datos/en-datos-asi-escuchan-musica-las-personas-en-el-mundo-636182
