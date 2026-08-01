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
        # Setup title and lecture lines
        title = "The Hook: Meet Rex the Rubber Dinosaur"
        lines = [
            "Meet Rex, a rubber dinosaur on a number line.",
            "As Rex moves, his body stretches or compresses.",
            "One step forward might stretch his tail three units.",
            "This isn't just a slope on a static graph.",
            "It's a local stretch factor at a specific point."
        ]
        self.setup_layout(title, lines)
        
        # Animation state trackers
        self.pos_tracker = ValueTracker(0.5) # x-position index (0 to 5 for col 1 to 6)
        self.body_stretch_tracker = ValueTracker(1.0)
        self.tail_stretch_tracker = ValueTracker(1.0)
        
        # === Animation for Lecture Line 1 ===
        # Meet Rex, a rubber dinosaur on a number line.
        self.lecture[0].set_color(YELLOW)
        
        # Number line covering grid cols 1 to 6 on Row C
        number_line = NumberLine(x_range=[0, 5, 1], length=5, include_tip=True, color=WHITE)
        # Resolved Issue 26: Reduced scale to avoid right edge clipping
        self.place_in_area(number_line, "C1", "C6", scale_factor=0.8)
        
        # Rex components (vector construction)
        rex_body = Ellipse(width=0.7, height=0.4, fill_opacity=1, color=BLUE_D)
        rex_head = Circle(radius=0.18, fill_opacity=1, color=BLUE_D)
        rex_tail = Triangle(fill_opacity=1, color=BLUE_D).scale(0.15).rotate(-PI/2)
        rex_eye = Dot(radius=0.03, color=BLACK)
        rex_legs = VGroup(
            Line(ORIGIN, DOWN*0.7, color=BLUE_D, stroke_width=4),
            Line(ORIGIN, DOWN*0.7, color=BLUE_D, stroke_width=4)
        ).arrange(RIGHT, buff=0.2)
        
        rex = VGroup(rex_body, rex_head, rex_tail, rex_eye, rex_legs)
        
        # Continuous updater for Rex's position and shape
        def update_rex(mob):
            pos = self.pos_tracker.get_value()
            b_stretch = self.body_stretch_tracker.get_value()
            t_stretch = self.tail_stretch_tracker.get_value()
            
            # Use grid coordinates for Row B (y=1.2)
            # Center of col 1 is 0.5, col 2 is 1.5, etc.
            grid_x = 0.5 + pos * 1
            grid_y = 1.2 
            mob.move_to([grid_x, grid_y, 0])
            
            # Update parts
            mob[0].stretch_to_fit_width(0.7 * b_stretch)
            mob[1].next_to(mob[0], RIGHT, buff=0)
            mob[2].stretch_to_fit_width(0.3 * t_stretch)
            mob[2].next_to(mob[0], LEFT, buff=0)
            mob[3].move_to(mob[1].get_center() + RIGHT*0.06 + UP*0.04)
            mob[4].move_to(mob[0].get_bottom() + DOWN*0.1)

        rex.add_updater(update_rex)
        self.add(rex)
        self.play(Create(number_line), FadeIn(rex))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # As Rex moves, his body stretches or compresses.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(
            self.pos_tracker.animate.set_value(2.0), # Moving towards center
            self.body_stretch_tracker.animate.set_value(1.5),
            run_time=2.5,
            rate_func=linear
        )
        self.play(
            self.pos_tracker.animate.set_value(3.5),
            self.body_stretch_tracker.animate.set_value(0.7),
            run_time=2.0,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # One step forward might stretch his tail three units.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Tail glow highlight
        glow = rex[2].copy().set_color("#00FF00").set_stroke(width=10, opacity=0.8)
        # Persistent mobject with updater
        glow.add_updater(lambda m: m.match_points(rex[2]).set_color("#00FF00").set_stroke(width=10, opacity=0.8))
        self.add(glow)
        
        self.play(
            self.pos_tracker.animate.set_value(4.5),
            self.tail_stretch_tracker.animate.set_value(3.5),
            self.body_stretch_tracker.animate.set_value(1.0),
            run_time=2.5
        )
        self.wait(1)
        self.remove(glow)

        # === Animation for Lecture Line 4 ===
        # This isn't just a slope on a static graph.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Create a small XY graph to contrast
        axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1],
            x_length=2.5, y_length=1.8,
            axis_config={"include_tip": False, "color": GREY}
        )
        graph = axes.plot(lambda x: 0.2*x**2, color=BLUE)
        graph_group = VGroup(axes, graph)
        # Resolved Issue 25: Moved down to avoid cluttering number line
        self.place_in_area(graph_group, "E4", "F6", scale_factor=0.8)
        
        self.play(FadeIn(graph_group))
        self.wait(2)
        self.play(FadeOut(graph_group))

        # === Animation for Lecture Line 5 ===
        # It's a local stretch factor at a specific point.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        stretch_label = Text("Stretch Factor", color="#FFFF00", font_size=24)
        # Resolved Issue 24: Changed to place_in_area with smaller scale to avoid clipping
        self.place_in_area(stretch_label, "A4", "A5", scale_factor=0.7)
        
        # Arrow pointing to the transformation point
        arrow = Arrow(UP, DOWN, color="#FFFF00", buff=0.1)
        arrow.add_updater(lambda m: m.put_start_and_end_on(stretch_label.get_bottom(), rex[0].get_top()))
        
        self.play(Write(stretch_label), Create(arrow))
        self.wait(3)

        # Transition cleanup
        self.play(FadeOut(rex, number_line, stretch_label, arrow))
