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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Why It Matters: Applications", ["Gaussian summation simplifies complex modeling.", "It helps in noise reduction.", "Essential for engineering systems."])
        
        # Assets: Icons/Visuals
        icon_heights = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        icon_scores = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/test.svg")
        bell_curve = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-2, 2], color="#FFFF00")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        icon_heights.set_color("#00FF00")
        icon_scores.set_color("#00FFFF")
        self.place_at_grid(icon_heights, 'B3', scale_factor=0.7)
        self.place_at_grid(icon_scores, 'B5', scale_factor=0.7)
        self.play(FadeIn(icon_heights), FadeIn(icon_scores))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.place_in_area(bell_curve, 'D3', 'F5', scale_factor=0.6)
        self.play(Create(bell_curve))
        self.play(
            icon_heights.animate.move_to(bell_curve.get_center()),
            icon_scores.animate.move_to(bell_curve.get_center())
        )
        self.wait(2)
