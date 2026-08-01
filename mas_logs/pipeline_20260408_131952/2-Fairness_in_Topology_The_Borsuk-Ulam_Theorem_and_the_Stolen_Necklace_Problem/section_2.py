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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title_text = "Prerequisite: The Intermediate Value Theorem (1D Case)"
        lecture_lines = [
            "Slide a knife along a necklace of one type.",
            "Alice's share changes continuously from zero to total.",
            "The Intermediate Value Theorem guarantees a perfect 50/50 split."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        RED_BEAD = "#FF0000"
        KNIFE_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(RED_BEAD)
        
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/necklace.svg
        necklace_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/necklace.svg")
        necklace_svg.set_color(RED_BEAD)
        necklace_group = VGroup(necklace_svg)
        
        # Issue 29: Position necklace group
        self.place_in_area(necklace_group, 'A3', 'B6', scale_factor=0.9)
        
        # Knife object
        knife = Line(
            start=necklace_svg.get_top() + UP * 0.2,
            end=necklace_svg.get_bottom() + DOWN * 0.2,
            color=KNIFE_COLOR,
            stroke_width=6
        )
        
        knife_tracker = ValueTracker(0)
        # Update knife to stay on the necklace path horizontally
        knife.add_updater(lambda m: m.move_to(
            interpolate(necklace_svg.get_left(), necklace_svg.get_right(), knife_tracker.get_value())
        ))

        self.play(Create(necklace_group))
        self.add(knife)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED_BEAD)
        
        # Issue 28: Plot group setup and positioning
        axes = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 100, 50],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": False, "font_size": 20},
            tips=False
        )
        x_label = Text("Cut Position", font_size=16).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Text("Alice's %", font_size=16).next_to(axes.y_axis, LEFT, buff=0.2).rotate(90 * DEGREES)
        plot_group = VGroup(axes, x_label, y_label)
        self.place_in_area(plot_group, 'C3', 'F6', scale_factor=0.8)
        
        # Plot line updater
        plot_line = always_redraw(lambda: axes.plot(
            lambda x: 100 * x, 
            x_range=[0, max(0.001, knife_tracker.get_value())], 
            color=RED_BEAD
        ))
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.add(plot_line)
        
        # Slide knife from 0 to 1
        self.play(knife_tracker.animate.set_value(1), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Show specific point at 50%
        self.play(knife_tracker.animate.set_value(0.5), run_time=2)
        
        target_point = axes.c2p(0.5, 50)
        highlight_dot = Dot(target_point, color=HIGHLIGHT_COLOR, radius=0.1)
        
        # Issue 30: Place 50% label at grid
        label_50 = Text("50%", font_size=18, color=HIGHLIGHT_COLOR)
        self.place_at_grid(label_50, 'D3', scale_factor=1.2)
        
        # Indicative lines
        h_line = axes.get_horizontal_line(target_point, color=HIGHLIGHT_COLOR)
        v_line = axes.get_vertical_line(target_point, color=HIGHLIGHT_COLOR)
        
        self.play(
            FadeIn(highlight_dot, scale=0.5),
            Create(h_line),
            Create(v_line),
            Write(label_50)
        )
        self.play(Indicate(highlight_dot, color=HIGHLIGHT_COLOR))
        
        self.wait(2)
