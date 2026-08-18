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
        self.setup_layout(
            "The Hook: The Speeding Cheetah", 
            [
                "Meet the world's fastest land animal: the cheetah.",
                "Calculating average speed over total distance is easy.",
                "But how fast is it at one exact moment?"
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Asset: cheetah.svg
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg", color=ORANGE, fill_opacity=1)
        self.place_at_grid(cheetah, "F1", scale_factor=0.3)
        
        cheetah_label = Text("Cheetah", font_size=16, color=ORANGE)
        cheetah_label.next_to(cheetah, UP, buff=0.1)

        self.play(DrawBorderThenFill(cheetah), Write(cheetah_label))
        
        # Move cheetah across the bottom to F6 (Issue 24: use F6 as anchor)
        target_pos = self.grid["F6"]
        self.play(
            cheetah.animate.move_to(target_pos),
            cheetah_label.animate.move_to(target_pos + UP * 0.4),
            run_time=3,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Create coordinate system
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 25, 5],
            x_length=4,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": True},
            tips=True
        )
        labels = axes.get_axis_labels(x_label="t", y_label="f(t)")
        
        # Plot curved position graph f(t) = t^2
        graph = axes.plot(lambda t: t**2, x_range=[0, 4.5], color=WHITE)
        
        graph_group = VGroup(axes, labels, graph)
        self.place_in_area(graph_group, "B1", "E4", scale_factor=0.8)
        
        self.play(Create(axes), Write(labels))
        self.play(Create(graph))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Asset: speed.svg (Issue 19)
        # Issue 25: scale 0.6 at A5
        speedo_base = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg")
        speedo_needle = Line(start=ORIGIN, end=[0.4, 0, 0], color=RED, stroke_width=4)
        speedo_center = Dot(color=WHITE, radius=0.05)
        speedo = VGroup(speedo_base, speedo_needle, speedo_center)
        self.place_at_grid(speedo, "A5", scale_factor=0.6)
        
        # Align needle to center of SVG base
        speedo_needle.move_to(speedo_base.get_center(), aligned_edge=LEFT)
        speedo_center.move_to(speedo_base.get_center())

        # Animation for needle (ValueTracker)
        needle_angle = ValueTracker(0)
        speedo_needle.add_updater(
            lambda m: m.set_angle(needle_angle.get_value()).move_to(speedo_base.get_center(), aligned_edge=LEFT)
        )
        
        self.play(FadeIn(speedo_base), Create(speedo_needle), GrowFromCenter(speedo_center))
        
        # Fluctuating needle
        self.play(needle_angle.animate.set_value(PI/3), run_time=0.5)
        self.play(needle_angle.animate.set_value(PI/6), run_time=0.5)
        self.play(needle_angle.animate.set_value(PI/2), run_time=0.5)
        self.play(needle_angle.animate.set_value(PI/4), run_time=0.5)

        # Highlight specific point on the curve
        point_t = 3.0
        point_coords = axes.c2p(point_t, point_t**2)
        highlight_circle = Circle(radius=0.1, color=YELLOW).move_to(point_coords)
        
        # Issue 23: moment_text area B5-B6, scale 0.7
        moment_text = Text("Speed at this exact moment?", font_size=18, color=YELLOW)
        self.place_in_area(moment_text, "B5", "B6", scale_factor=0.7)
        
        # Draw a line from point to text for clarity
        pointer_line = Arrow(start=moment_text.get_bottom(), end=highlight_circle.get_top(), color=YELLOW, buff=0.1)

        self.play(Create(highlight_circle))
        self.play(Write(moment_text), Create(pointer_line))
        self.wait(2)

        # Cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
