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
        # Data from storyboard
        title_text = "Prerequisite Knowledge: The Two Sides of Motion"
        lecture_lines = [
            "Meet Swiftie the Snail moving along a ruler.",
            "If we know position, can we find speed?",
            "If we know speed, can we find distance?"
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        RULER_COLOR = "#FFFFFF"
        POSITION_COLOR = "#ADD8E6"
        SPEED_COLOR = "#90EE90"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # Use color changes for lecture lines
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Ruler creation
        ruler_line = Line(LEFT * 2, RIGHT * 2, color=RULER_COLOR)
        ticks = VGroup(*[
            Line(
                start=ruler_line.point_from_proportion(i/5) + DOWN * 0.1,
                end=ruler_line.point_from_proportion(i/5) + UP * 0.1,
                color=RULER_COLOR
            ) for i in range(6)
        ])
        ruler = VGroup(ruler_line, ticks)
        
        # Swiftie (Snail) - Simple vector representation
        swiftie_body = Ellipse(width=0.4, height=0.2, color=WHITE, fill_opacity=1)
        swiftie_eye = Dot(radius=0.03, color=BLACK).move_to(swiftie_body.get_right() + UP*0.05 + LEFT*0.05)
        swiftie = VGroup(swiftie_body, swiftie_eye)
        swiftie_label = Text("Swiftie", font_size=16, color=WHITE)
        
        # Snail Group (Snail + Label)
        snail_group = VGroup(swiftie, swiftie_label)
        snail_group[1].next_to(snail_group[0], UP, buff=0.1)
        
        # Full scene group for positioning as per Issue 35
        # The critic suggested place_in_area for snail_group (including ruler)
        master_snail_group = VGroup(ruler, snail_group)
        self.place_in_area(master_snail_group, 'E2', 'F6', scale_factor=0.9)

        # Re-position snail relative to the scaled ruler
        snail_group.move_to(ruler_line.get_left() + UP * 0.2)

        self.play(Create(ruler), FadeIn(snail_group))
        
        # Movement tracker for snail along ruler
        path_tracker = ValueTracker(0)
        
        # Updater for snail movement
        snail_group.add_updater(lambda m: m.move_to(ruler_line.point_from_proportion(path_tracker.get_value()) + UP * 0.2))
        
        # Initial movement
        self.play(path_tracker.animate.set_value(0.3), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(POSITION_COLOR)
        )
        
        # Graph - Issue 36 fix: Move axes to 'B2' to 'D6' to reduce clutter
        axes = Axes(
            x_range=[0, 1.1, 0.2],
            y_range=[0, 1.1, 0.2],
            x_length=4,
            y_length=2.5,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        self.place_in_area(axes, 'B2', 'D6', scale_factor=0.8)
        
        pos_dot = Dot(color=POSITION_COLOR)
        pos_label = Text("Position", color=POSITION_COLOR, font_size=18)
        
        # Updater for position dot on graph (showing constant velocity)
        pos_dot.add_updater(lambda m: m.move_to(axes.c2p(path_tracker.get_value(), path_tracker.get_value())))
        pos_label.add_updater(lambda m: m.next_to(pos_dot, UP, buff=0.1))

        self.play(Create(axes), FadeIn(pos_dot), FadeIn(pos_label))
        
        # Continue movement
        self.play(path_tracker.animate.set_value(0.6), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(SPEED_COLOR)
        )
        
        # Speed vector (arrow growing from snail)
        speed_arrow = Arrow(
            start=LEFT, 
            end=RIGHT, 
            color=SPEED_COLOR, 
            buff=0, 
            stroke_width=5,
            max_tip_length_to_length_ratio=0.25
        ).scale(0.5)
        speed_label = Text("Speed", color=SPEED_COLOR, font_size=18)
        
        # Updater for speed arrow and label relative to the snail
        speed_arrow.add_updater(lambda m: m.next_to(swiftie, RIGHT, buff=0.1))
        speed_label.add_updater(lambda m: m.next_to(speed_arrow, UP, buff=0.1))
        
        self.play(GrowArrow(speed_arrow), FadeIn(speed_label))
        
        # Final movement to end
        self.play(path_tracker.animate.set_value(1.0), run_time=3, rate_func=linear)
        self.wait(2)
