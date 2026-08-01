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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Differentiation and integration are two sides of one coin.",
            "The rate of area growth equals the graph's height.",
            "One operation undoes the effect of the other.",
            "This link is the Fundamental Theorem of Calculus.",
            "It bridges the gap between change and accumulation."
        ]
        self.setup_layout("The Grand Connection: The Fundamental Theorem", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Morph a large white 'Σ' symbol into a magenta '∫' symbol #FF00FF. Then self.wait(2.0).
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        
        # Positioning adjustment as per Issue 32 (C3-E4 for better vertical balance)
        sigma = MathTex(r"\Sigma", color="#FFFFFF", font_size=120)
        integral = MathTex(r"\int", color="#FF00FF", font_size=120)
        
        self.place_in_area(sigma, "C3", "E4", scale_factor=0.8)
        self.place_in_area(integral, "C3", "E4", scale_factor=0.8)
        
        self.play(Write(sigma))
        self.wait(1.0)
        self.play(ReplacementTransform(sigma, integral))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Show a graph where a moving white vertical line #FFFFFF highlights the 'Height' at each point. Then self.wait(2.0).
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        self.play(FadeOut(integral))
        
        # Graph setup - Restricted area as per Issue 31 (B2-E6 to avoid Row F overlap)
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"color": "#FFFFFF", "include_tip": False}
        )
        
        func = lambda x: 0.2 * x**2 + 0.5
        graph = axes.plot(func, x_range=[0, 3.5], color="#00FFFF")
        
        graph_group = VGroup(axes, graph)
        self.place_in_area(graph_group, "B2", "E6", scale_factor=0.8)
        
        self.play(Create(axes), Create(graph))
        
        # Value tracker for movement
        x_tracker = ValueTracker(1)
        
        # Height line (persistent + updater)
        height_line = Line(
            axes.c2p(1, 0),
            axes.c2p(1, func(1)),
            color="#FFFFFF",
            stroke_width=4
        )
        height_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(x_tracker.get_value(), 0),
            axes.c2p(x_tracker.get_value(), func(x_tracker.get_value()))
        ))
        
        # Label following the line
        height_label = MathTex("f(x)", color="#FFFFFF").scale(0.7)
        height_label.add_updater(lambda m: m.next_to(height_line, UP, buff=0.1))
        
        self.play(Create(height_line), Write(height_label))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # One operation undoes the effect of the other. (Area growth / Integration)
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        
        # Area growth - using always_redraw for the dynamic polygon shape
        area = always_redraw(lambda: axes.get_area(
            graph, x_range=[0, x_tracker.get_value()], color="#FF00FF", opacity=0.4
        ))
        
        self.add(area)
        self.play(x_tracker.animate.set_value(3), run_time=4)
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # Use a green #00FF00 arrow to link the moving 'Height' to the growth of the 'Total Area'. Then self.wait(2.0).
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        
        # Area label positioned in Row F as per Issue 33
        area_label = Text("Total Area", color="#00FF00")
        self.place_in_area(area_label, 'F3', 'F5', scale_factor=0.8)
        
        # Arrow connecting height tip to label
        curr_x = x_tracker.get_value()
        arrow = Arrow(
            start=axes.c2p(curr_x, func(curr_x)),
            end=area_label.get_top(),
            color="#00FF00",
            max_tip_length_to_length_ratio=0.15 # L020
        )
        
        self.play(GrowArrow(arrow), Write(area_label))
        self.play(Indicate(arrow)) # L004
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # It bridges the gap between change and accumulation.
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        
        # Final connection highlight
        self.play(Indicate(height_line), Indicate(area))
        self.wait(2.0)
        
        # Reset and conclude
        self.play(
            FadeOut(graph_group),
            FadeOut(height_line),
            FadeOut(height_label),
            FadeOut(area),
            FadeOut(arrow),
            FadeOut(area_label),
            self.lecture.animate.set_color(WHITE)
        )
        self.wait(2.0)
