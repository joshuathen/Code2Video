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
        self.setup_layout("Prerequisite: The Basis Vectors", [
            "3D space uses unit vectors î, ĵ, k̂.",
            "Every vector is a combination of these.",
            "Imagine these as anchors for space."
        ])
        
        # Define objects
        # Axes for grid - positioned to avoid lecture text
        axes = Axes(x_range=[-1, 3, 1], y_range=[-1, 3, 1], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'A4', 'F6', scale_factor=0.6)
        
        # Corrected grid implementation - positioned to avoid lecture text
        grid = NumberPlane(x_range=[-1, 3, 1], y_range=[-1, 3, 1]).set_color("#A9A9A9")
        self.place_in_area(grid, 'A4', 'F6', scale_factor=0.3)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/anchor.svg
        anchor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/anchor.svg")
        self.place_at_grid(anchor_icon, 'B5', scale_factor=0.15)
        
        # Create vectors
        i_hat = Arrow(start=ORIGIN, end=axes.c2p(1, 0), color="#FFD700", buff=0)
        j_hat = Arrow(start=ORIGIN, end=axes.c2p(0, 1), color="#FFD700", buff=0)
        i_label = MathTex(r"\hat{i}", color="#FFD700").next_to(i_hat.get_end(), DOWN)
        j_label = MathTex(r"\hat{j}", color="#FFD700").next_to(j_hat.get_end(), LEFT)

        # === Animation for Lecture Line 1 ===
        self.play(Create(grid))
        self.play(Create(i_hat), Create(j_hat), Write(i_label), Write(j_label), FadeIn(anchor_icon))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.play(i_hat.animate.set_color("#FFFFFF"), j_hat.animate.set_color("#FFFFFF"))
        self.play(Indicate(i_hat), Indicate(j_hat))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#A9A9A9")
        self.wait(1)
