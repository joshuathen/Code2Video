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
        self.setup_layout("Simulation & Intervention: Flattening the Curve", 
                          ["Interventions modify transmission parameters.", 
                           "Social distancing reduces contact rates.", 
                           "[Asset: Flattened_Curve_Graph] visualizes the change."])
        
        # Assets / Graph elements
        axes = Axes(x_range=[0, 10, 1], y_range=[0, 5, 1], axis_config={"include_tip": False}).scale(0.6)
        # Placeholder curves
        curve_orig = axes.plot(lambda x: 4 * (x/5) * np.exp(1 - x/5), color=RED)
        curve_flat = axes.plot(lambda x: 2 * (x/5) * np.exp(1 - x/5), color=GREEN)
        
        # Applying the fix from issues 29, 30, 31, 38
        self.place_in_area(axes, 'B4', 'E6', scale_factor=0.85)
        
        # Asset placeholders (as requested in instruction 6 & issue 19)
        # Using SVG placeholder icons since the specific files in path might not exist
        icon_orig = Square(side_length=0.5, color=RED).set_fill(RED, opacity=0.5)
        self.place_at_grid(icon_orig, 'B5')
        icon_flat = Square(side_length=0.5, color=GREEN).set_fill(GREEN, opacity=0.5)
        self.place_at_grid(icon_flat, 'E5')
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), Create(curve_orig), FadeIn(icon_orig))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        intervention_line = DashedLine(axes.c2p(0, 2), axes.c2p(10, 2), color=BLUE)
        self.play(Create(intervention_line))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Transform(curve_orig, curve_flat), FadeIn(icon_flat))
        self.wait(2)
