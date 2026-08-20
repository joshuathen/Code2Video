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
            "Vectors are arrows with magnitude and direction.",
            "Vector addition uses the tip-to-tail method.",
            "Scalar multiplication scales the arrow's length."
        ]
        self.setup_layout("Prerequisite Review: Vectors as Arrows", lecture_lines)
        
        # Placeholder image as assets are none.svg
        # In a real scenario, use ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        
        v_vector = Arrow(start=ORIGIN, end=RIGHT*1.5 + UP*1.0, color="#FF5733")
        self.place_in_area(v_vector, 'C3', 'E5', scale_factor=0.9)
        
        v_label = MathTex(r"\\vec{v}", color=WHITE)
        self.place_at_grid(v_label, 'C3', scale_factor=0.7)
        
        self.play(Create(v_vector), Write(v_label))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        
        u_vector = Arrow(start=ORIGIN, end=RIGHT*1.0 + DOWN*0.5, color="#33FF57")
        # u_vector start position relative to v_vector end as per "tip-to-tail"
        u_vector.shift(v_vector.get_end())
        
        u_label = MathTex(r"\\vec{u}", color="#33FF57")
        self.place_at_grid(u_label, 'D4', scale_factor=0.7)
        
        self.play(Create(u_vector), Write(u_label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        
        new_v = Arrow(start=ORIGIN, end=(RIGHT*1.5 + UP*1.0)*2, color="#3357FF")
        self.place_in_area(new_v, 'C3', 'E5', scale_factor=0.9)
        
        self.play(Transform(v_vector, new_v))
        self.play(FadeOut(u_vector), FadeOut(u_label))
        self.wait(1)
