from manim import *
import numpy as np

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
            "Meet Leo, sunbathing on a warm, desert rock.",
            "The rock's temperature varies by position and time.",
            "Ordinary equations track change over just one variable.",
            "But PDEs track change across space and time simultaneously.",
            "They model how heat flows through the entire rock."
        ]
        self.setup_layout("The Hook: Leo the Lizard and the Hot Rock", lecture_lines)

        # Colors
        ROCK_GREY = "#808080"
        HEAT_START = "#FF4500"
        HEAT_END = "#FFD700"
        RIPPLE_COLOR = "#FF0000"
        HIGHLIGHT_COLOR = "#FFFF00"
        LIZARD_GREEN = "#32CD32"
        ODE_BLUE = "#58C4DD"
        PDE_TEAL = "#5CD0B3"

        # === Animation for Lecture Line 1 ===
        # Create a simple lizard shape on a grey rectangle.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        rock = Rectangle(width=4, height=3, fill_color=ROCK_GREY, fill_opacity=1.0, stroke_color=WHITE)
        self.place_in_area(rock, 'A2', 'F5')
        
        leo_body = Ellipse(width=1.2, height=0.4, fill_color=LIZARD_GREEN, fill_opacity=1.0).rotate(PI/12)
        leo_head = Ellipse(width=0.4, height=0.3, fill_color=LIZARD_GREEN, fill_opacity=1.0).next_to(leo_body, RIGHT, buff=-0.1)
        leo_tail = Triangle(fill_color=LIZARD_GREEN, fill_opacity=1.0).scale(0.2).rotate(-PI/2).next_to(leo_body, LEFT, buff=-0.1)
        leo = VGroup(leo_body, leo_head, leo_tail)
        self.place_in_area(leo, 'C3', 'D4')
        
        self.play(FadeIn(rock), FadeIn(leo))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color the rock with a gradient #FF4500 to #FFD700.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        self.play(rock.animate.set_fill(color=[HEAT_START, HEAT_END]))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Show a simple line graph moving forward in time.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        self.play(rock.animate.set_opacity(0.15), leo.animate.set_opacity(0.15))
        
        axes = Axes(
            x_range=[0, 5, 1], y_range=[0, 3, 1],
            axis_config={"include_tip": True, "color": WHITE},
            x_length=3.5, y_length=2.5
        )
        self.place_in_area(axes, 'B2', 'E5')
        
        sin_graph = axes.plot(lambda x: 1.5 + 0.5 * np.sin(x), x_range=[0, 5], color=ODE_BLUE)
        graph_label = Text("T(t)", font_size=20, color=ODE_BLUE)
        self.place_at_grid(graph_label, 'B5')
        
        self.play(Create(axes), Create(sin_graph), Write(graph_label))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Expand the graph into a 2D grid pulsing with color.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        grid_2d = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=3.5, y_length=3.5,
            background_line_style={"stroke_color": ODE_BLUE, "stroke_width": 2}
        )
        self.place_in_area(grid_2d, 'B2', 'E5')
        
        pde_label = Text("T(x, y, t)", font_size=20, color=PDE_TEAL)
        self.place_at_grid(pde_label, 'A5')
        
        pulse_tracker = ValueTracker(0)
        grid_2d.add_updater(
            lambda m: m.set_color(interpolate_color(ODE_BLUE, PDE_TEAL, (np.sin(pulse_tracker.get_value() * PI) + 1) / 2))
        )
        
        self.play(
            FadeOut(axes), FadeOut(sin_graph), FadeOut(graph_label),
            FadeIn(grid_2d), Write(pde_label)
        )
        self.play(pulse_tracker.animate.set_value(4), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show heat ripples #FF0000 moving outward from a center point.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        self.play(FadeOut(pde_label), FadeOut(grid_2d), rock.animate.set_opacity(1.0), leo.animate.set_opacity(1.0))
        
        ripple_radius = ValueTracker(0.1)
        ripple_opacity = ValueTracker(1.0)
        
        ripple_mobj = Circle(radius=0.1, color=RIPPLE_COLOR, stroke_width=4)
        ripple_mobj.add_updater(lambda m: m.set_width(ripple_radius.get_value() * 2 if ripple_radius.get_value() > 0 else 0.01))
        ripple_mobj.add_updater(lambda m: m.set_stroke(opacity=ripple_opacity.get_value()))
        
        self.place_in_area(ripple_mobj, 'C3', 'D4')
        self.add(ripple_mobj)
        
        for _ in range(3):
            ripple_radius.set_value(0.1)
            ripple_opacity.set_value(1.0)
            self.play(
                ripple_radius.animate.set_value(2.5),
                ripple_opacity.animate.set_value(0),
                run_time=1.2,
                rate_func=linear
            )
            
        self.wait(2)
        self.lecture[4].set_color(WHITE)
