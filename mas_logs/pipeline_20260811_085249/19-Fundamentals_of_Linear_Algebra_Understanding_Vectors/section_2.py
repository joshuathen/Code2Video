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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Vector Addition: The Head-to-Tail Rule", [
            "Vector addition uses the head-to-tail rule.",
            "Place the second tail at the first head.",
            "The result is a new, single vector."
        ])
        
        # Assets placeholders
        # NOTE: Using SVGMobject for asset references even if the path is none.svg
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Create vectors
        u = Arrow(start=ORIGIN, end=RIGHT*1.5 + UP*0.5, color=BLUE)
        v = Arrow(start=ORIGIN, end=LEFT*0.5 + UP*1.2, color=RED)
        
        u_label = MathTex(r"\\vec{u}", color=BLUE).next_to(u.get_center(), UP, buff=0.1)
        v_label = MathTex(r"\\vec{v}", color=RED).next_to(v.get_center(), LEFT, buff=0.1)
        
        vectors_group = VGroup(u, u_label, v, v_label, icon1)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(vectors_group, 'B4', 'E6', scale_factor=0.8)
        self.place_at_grid(icon1, 'B6', scale_factor=0.5)
        self.play(Create(u), Write(u_label), Create(v), Write(v_label), FadeIn(icon1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        
        # Calculate vector v's new position: tail at u's head
        v_target_pos = u.get_end()
        v_vector_shift = v_target_pos - v.get_start()
        
        self.play(
            v.animate.shift(v_vector_shift),
            v_label.animate.shift(v_vector_shift)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        resultant = Arrow(start=ORIGIN, end=v.get_end(), color=GREEN)
        resultant_label = MathTex(r"\\vec{u}+\\vec{v}", color=GREEN)
        self.place_at_grid(resultant_label, 'F3', scale_factor=0.7)
        
        self.place_at_grid(icon2, 'F5', scale_factor=0.5)
        
        self.play(Create(resultant), Write(resultant_label), FadeIn(icon2))
        self.wait(2)
