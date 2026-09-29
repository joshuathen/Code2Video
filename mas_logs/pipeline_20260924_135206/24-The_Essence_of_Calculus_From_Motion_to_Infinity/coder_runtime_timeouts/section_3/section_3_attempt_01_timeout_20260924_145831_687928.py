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
            "Derivatives measure the rate of change precisely.",
            "The speedometer shows instantaneous speed changes.",
            "The derivative represents the curve's slope."
        ]
        self.setup_layout("Differentiation: Measuring Change", lecture_lines)
        
        # --- Animation components ---
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.25 * x**3, color="#FF4500")
        
        # Area for plot
        self.place_in_area(axes, "A1", "F6", scale_factor=0.6)
        axes.shift(RIGHT * 1) # Adjust to center in the right half
        curve.match_y(axes)
        curve.match_x(axes)
        
        point = ValueTracker(1.0)
        
        tangent = always_redraw(lambda: TangentLine(
            curve, alpha=point.get_value(), length=2, color="#FFFF00"
        ))
        
        slope_label = always_redraw(lambda: Text(
            f"Slope: {0.75 * point.get_value()**2:.2f}", 
            font_size=20, 
            color="#FFFFFF"
        ))
        self.place_at_grid(slope_label, "A3", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF4500"))
        self.play(Create(axes), Create(curve))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(Create(tangent))
        self.play(point.animate.set_value(3.0), run_time=3)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(FadeIn(slope_label))
        self.play(point.animate.set_value(1.0), run_time=2)
