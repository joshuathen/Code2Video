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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis & Application", [
            "Matrix is a spatial instruction set.",
            "It deforms space mathematically.",
            "Foundation of graphics and data analysis."
        ])
        
        # Assets
        monitor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg", color="#FFFFFF")
        smartphone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg", color="#00FF00")
        
        # Other objects
        matrix_icon = MathTex(r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}", color="#FF00FF")
        effect_label = Text("Composite Transformation", font_size=20, color="#FF00FF")
        wireframe = Square(side_length=1.5, color="#00FF00")
        
        composite_group = VGroup(matrix_icon, effect_label)
        
        # === Animation for Lecture Line 1 ===
        # "Matrix is a spatial instruction set."
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(monitor, 'B5', scale_factor=0.6)
        self.play(FadeIn(monitor))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "It deforms space mathematically."
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.place_in_area(composite_group, 'B3', 'E5', scale_factor=0.85)
        self.play(Write(composite_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Foundation of graphics and data analysis."
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.place_at_grid(smartphone, 'E5', scale_factor=0.5)
        self.place_at_grid(wireframe, 'D4', scale_factor=0.8)
        self.play(FadeIn(smartphone), Create(wireframe))
        self.play(wireframe.animate.scale(0.8).rotate(PI/4))
        self.wait(2)
