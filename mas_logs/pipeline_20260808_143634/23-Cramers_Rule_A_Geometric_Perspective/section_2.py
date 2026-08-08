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
        self.setup_layout("The Geometric Setup", ["A linear system is a vector equation.", "We scale columns to reach destination b.", "Find weights x and y to solve."])
        
        # Assets
        scales = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scales.svg")
        weights = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/weights.svg")
        
        # Mobjects
        system = MathTex(r"A \mathbf{x} = \mathbf{b}", font_size=48)
        vec_v1 = Vector([1, 2], color=BLUE)
        vec_v2 = Vector([2, -0.5], color=GREEN)
        vec_b = Vector([3, 1.5], color=YELLOW)
        
        label_v1 = MathTex(r"\mathbf{v_1}", color=BLUE, font_size=32)
        label_v2 = MathTex(r"\mathbf{v_2}", color=GREEN, font_size=32)
        label_b = MathTex(r"\mathbf{b}", color=YELLOW, font_size=32)
        
        # Helper to position label
        label_v1.next_to(vec_v1.get_end(), UP, buff=0.1)
        label_v2.next_to(vec_v2.get_end(), DOWN, buff=0.1)
        label_b.next_to(vec_b.get_end(), RIGHT, buff=0.1)
        
        # Combined group for geometric elements
        geometric_group = VGroup(vec_v1, vec_v2, vec_b, label_v1, label_v2, label_b, weights)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(scales, "C3", scale_factor=0.5)
        self.place_at_grid(system, "B3", scale_factor=0.9)
        self.play(Write(system), FadeIn(scales))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_in_area(geometric_group, "C2", "E5", scale_factor=0.85)
        self.play(FadeIn(geometric_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
