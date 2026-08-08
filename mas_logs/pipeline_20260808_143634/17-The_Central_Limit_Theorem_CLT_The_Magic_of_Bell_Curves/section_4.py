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
        lecture_lines = ["The magic number is thirty.", "Sample size affects the curve's shape.", "Larger samples reduce standard error.", "The Bell Curve becomes more precise.", "Precision increases with more data points."]
        self.setup_layout("The 'Sample Size' Rule", lecture_lines)
        
        # Assets
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        
        # Mobjects
        text_label = Text("Bigger samples reveal the true shape.", font_size=24, color=WHITE)
        self.place_at_grid(text_label, 'A1', scale_factor=0.9)
        self.place_at_grid(magnifying_glass, 'B1', scale_factor=0.3)
        
        # N=5 Curve
        def get_normal_curve(sigma):
            return FunctionGraph(
                lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2),
                x_range=[-3, 3], color=WHITE
            )
            
        curve_n5 = get_normal_curve(sigma=1.2)
        curve_n30 = get_normal_curve(sigma=0.5)
        
        # Applying layouts based on critiques
        self.place_in_area(curve_n5, 'B3', 'E5', scale_factor=0.6)
        self.place_in_area(curve_n30, 'B3', 'E5', scale_factor=0.6)
        
        # Position scale icon
        self.place_at_grid(scale_icon, 'F6', scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"), Write(text_label), FadeIn(magnifying_glass))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"), Create(curve_n5))
        curve_n5.set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"), Transform(curve_n5, curve_n30), FadeIn(scale_icon))
        curve_n5.set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.wait(1)
