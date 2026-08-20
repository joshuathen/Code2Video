from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Vectors start as arrows in space.",
            "They represent coordinate pairs.",
            "Now, vectors become generic objects.",
            "They must follow specific rules.",
            "Mathematics generalizes these structures."
        ]
        self.setup_layout("The Bridge: From Concrete to Abstract", lecture_lines)
        
        # Mobjects
        bicycle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bicycle.svg", color=WHITE)
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color="#FFD700")
        coords = MathTex(r"\\begin{bmatrix} x \\\\ y \\end{bmatrix}", color=WHITE)
        abstract_v = Tex(r"v", font_size=72, color="#FFD700")
        
        label_concrete = Text("Concrete", font_size=20, color=WHITE)
        label_abstract = Text("Abstract", font_size=20, color="#FFD700")
        connection = Line(start=ORIGIN, end=RIGHT*2, color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(bicycle, 'B2', scale_factor=0.3)
        self.place_at_grid(label_concrete, 'A2', scale_factor=0.6)
        self.play(FadeIn(bicycle), Write(label_concrete))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(coords, 'B4', scale_factor=0.8)
        self.play(ReplacementTransform(bicycle.copy(), coords))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.place_at_grid(abstract_v, 'E3', scale_factor=1.0)
        self.place_at_grid(label_abstract, 'D2', scale_factor=0.6)
        self.play(FadeIn(abstract_v), Write(label_abstract))

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]))
        self.place_in_area(connection, 'B3', 'C3', scale_factor=0.7)
        self.play(Create(connection))

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4]))
        self.place_at_grid(compass, 'E4', scale_factor=0.3)
        self.play(Flash(abstract_v), FadeIn(compass))
        self.play(self.lecture[0].animate.set_color(WHITE),
                  self.lecture[1].animate.set_color(WHITE),
                  self.lecture[2].animate.set_color("#FFD700"),
                  self.lecture[3].animate.set_color("#00FFFF"),
                  self.lecture[4].animate.set_color(WHITE))
