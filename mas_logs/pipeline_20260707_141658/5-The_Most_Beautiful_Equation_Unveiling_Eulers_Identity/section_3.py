from manim import *
import numpy as np

# Rate function for exponential growth animation
def exponential_rate_manual(t):
    # Maps linear t [0,1] to an exponential curve to show "speed increasing"
    k = 2
    return (np.exp(k * t) - 1) / (np.exp(k) - 1)

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
        # Data from storyboard
        title_text = "Prerequisite 2: 'e' as Constant Growth"
        lecture_lines = [
            "- The constant e is the base of continuous growth.",
            "- Growth rate is always proportional to the current value.",
            "- This creates a curve expanding faster and faster."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Initialize lecture lines to a dimmed color to show "highlighting" later
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1: Display the number 'e' with its decimal expansion (White #FFFFFF)
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        e_val = Text("e ≈ 2.71828...", color="#FFFFFF", font_size=32)
        # Resolving Issue 30: Placing e_val in area A2-A5
        self.place_in_area(e_val, "A2", "A5", scale_factor=0.8)
        
        self.play(Write(e_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2: Animate a bar growing upwards exponentially (Yellow #FFFFE0)
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#FFFFE0")
        )
        
        # Visualizing proportionality
        prop_logic = Text("Growth Rate = Current Value", color="#FFFFE0", font_size=28)
        # Resolving Issue 31: Placing prop_logic in area B2-B5
        self.place_in_area(prop_logic, "B2", "B5", scale_factor=0.8)
        
        # Bar animation
        bar = Rectangle(height=0.1, width=0.8, fill_opacity=1, color="#FFFFE0")
        self.place_at_grid(bar, "F4")
        
        growth_tracker = ValueTracker(0.1)
        bar.add_updater(lambda m: m.stretch_to_fit_height(max(0.01, growth_tracker.get_value()), about_edge=DOWN))
        
        self.play(FadeIn(prop_logic), Create(bar))
        # Use exponential rate function to show speed increasing as it gets taller
        self.play(growth_tracker.animate.set_value(3.0), run_time=3, rate_func=exponential_rate_manual)
        self.wait(1)
        bar.clear_updaters()

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3: Curve expanding (Pink #FFC0CB)
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#FFC0CB")
        )
        
        # Plotting the curve e^x
        axes = Axes(
            x_range=[0, 2.5, 1],
            y_range=[0, 10, 2],
            x_length=4.0,
            y_length=3.0,
            axis_config={"include_tip": True, "color": "#FFC0CB"}
        )
        # Resolving Issue 32: Placing axes in area C2-F6
        self.place_in_area(axes, "C2", "F6", scale_factor=0.9)
        
        graph = axes.plot(lambda x: np.exp(x), x_range=[0, 2.2], color="#FFC0CB")
        graph_label = Text("e^x", font_size=24, color="#FFC0CB")
        # Position label relative to end of curve
        graph_label.move_to(axes.c2p(2.2, np.exp(2.2)) + UR * 0.3)
        
        # Clean up screen for the graph
        self.play(FadeOut(bar), FadeOut(prop_logic), FadeOut(e_val))
        
        self.play(Create(axes), Create(graph))
        self.play(Write(graph_label))
        
        # Flash height and growth rate indicator to show they are equal
        target_x = 1.3
        target_y = np.exp(target_x)
        point = axes.c2p(target_x, target_y)
        
        height_line = Line(axes.c2p(target_x, 0), point, color=WHITE, stroke_width=4)
        growth_indicator = Arrow(point, point + UP * 1.0, color="#FFC0CB", buff=0)
        
        height_label = Text(f"Value: {target_y:.1f}", font_size=16, color=WHITE).next_to(height_line, LEFT, buff=0.1)
        growth_label = Text(f"Growth: {target_y:.1f}", font_size=16, color="#FFC0CB").next_to(growth_indicator, RIGHT, buff=0.1)
        
        self.play(Create(height_line), Write(height_label))
        self.play(GrowArrow(growth_indicator), Write(growth_label))
        self.wait(0.5)
        # Visual flash/indicate
        self.play(Indicate(height_line, color=WHITE), Indicate(growth_indicator, color="#FFC0CB"))
        
        self.wait(2)
        
        # Reset lecture line highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
