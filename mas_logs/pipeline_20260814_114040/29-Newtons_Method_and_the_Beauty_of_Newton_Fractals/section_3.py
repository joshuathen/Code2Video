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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Extend root finding to complex numbers.",
            "Polynomials like z^3-1 have roots.",
            "Which root will z_0 reach?"
        ]
        self.setup_layout("Transitioning to Complex Planes", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Extend root finding to complex numbers.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        grid = ComplexPlane().add_coordinates()
        grid.set_stroke(color="#444444", width=1)
        self.place_in_area(grid, "A4", "F6", scale_factor=0.6)
        self.play(Create(grid))
        self.lecture[0].set_color("#444444")
        
        # === Animation for Lecture Line 2 ===
        # Polynomials like z^3-1 have roots.
        formula = MathTex("f(z) = z^3 - 1").set_color("#FFFFFF")
        self.place_at_grid(formula, "C3", scale_factor=0.7)
        self.play(Write(formula))
        self.lecture[1].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 3 ===
        # Which root will z_0 reach?
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        z0 = Dot(point=self.grid["E4"], color="#FF00FF")
        z0_label = MathTex("z_0").next_to(z0, UP).set_color("#FF00FF")
        target = Dot(point=self.grid["D5"], color="#00FFFF")
        root_label = Text("root", font_size=24).set_color("#00FFFF")
        self.place_at_grid(root_label, "D5", scale_factor=0.6)
        root_label.next_to(target, DOWN)
        
        path = CurvedArrow(z0.get_center(), target.get_center(), angle=-TAU/8)
        self.place_in_area(path, "C3", "F6", scale_factor=0.7)
        
        self.play(FadeIn(z0), Write(z0_label))
        self.play(Create(path))
        self.play(FadeIn(target), Write(root_label))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
