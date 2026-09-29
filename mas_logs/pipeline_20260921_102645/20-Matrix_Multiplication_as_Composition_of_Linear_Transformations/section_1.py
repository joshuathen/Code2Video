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
        self.setup_layout("Prerequisite Review: The Geometric Meaning of a Matrix", 
                          ["A matrix is a machine transforming space.", 
                           "Columns reveal where basis vectors land.", 
                           "Example: A rotation matrix shifts basis."])
        
        # Setup Axes
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True})
        axes = self.place_in_area(axes, "A1", "F6", scale_factor=0.6)
        i_hat = Vector([1, 0], color=WHITE)
        j_hat = Vector([0, 1], color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), FadeIn(i_hat), FadeIn(j_hat))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color(WHITE))
        # Represent movement
        new_i = Vector([0, 1], color=BLUE)
        new_j = Vector([-1, 0], color=RED)
        self.play(i_hat.animate.become(new_i), j_hat.animate.become(new_j))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color(GOLD))
        # Using placeholder asset as requested (even if generic/none.svg)
        # Note: The asset path provided was /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Since it is a placeholder, we use a simple Dot to represent the icon
        icon = Dot(color=GOLD)
        self.place_at_grid(icon, "C3", scale_factor=2.0)
        self.play(FadeIn(icon))
        
        self.wait(2)
