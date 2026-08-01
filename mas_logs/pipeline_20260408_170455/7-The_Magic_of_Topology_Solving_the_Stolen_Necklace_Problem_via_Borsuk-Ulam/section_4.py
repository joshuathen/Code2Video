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
        # Setup
        lecture_lines = [
            'We can model the necklace as a continuous loop.',
            'A single cut point is like a position on a circle.',
            'Let f(x) be the amount of beads Thief A gets.',
            'As we move the cut, this share changes continuously.',
            'Multiple bead types require higher-dimensional spheres and mappings.'
        ]
        self.setup_layout("Connecting the Dots: Beads to Functions", lecture_lines)
        
        # Colors
        RED_COLOR = "#FF5555"
        BLUE_COLOR = "#5555FF"
        HIGHLIGHT_COLOR = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create linear necklace
        necklace_line = Line(start=LEFT, end=RIGHT, color=WHITE).set_length(4)
        beads = VGroup(*[
            Dot(color=RED_COLOR if i % 2 == 0 else BLUE_COLOR, radius=0.1)
            for i in range(12)
        ])
        for i, bead in enumerate(beads):
            bead.move_to(necklace_line.point_from_proportion(i / 11))
        
        linear_necklace = VGroup(necklace_line, beads)
        # Issue 35 fix: Changed B1-C4 to B2-D4
        self.place_in_area(linear_necklace, 'B2', 'D4', scale_factor=1.0)
        self.play(FadeIn(linear_necklace))
        self.wait(1)
        
        # Bend necklace into a circle
        necklace_circle = Circle(radius=1.2, color=WHITE)
        # Issue 35 fix: Changed B1-D3 to B2-D4
        self.place_in_area(necklace_circle, 'B2', 'D4', scale_factor=1.0)
        
        # Create circle beads
        circle_beads = VGroup(*[
            Dot(color=beads[i].get_color(), radius=0.1)
            for i in range(len(beads))
        ])
        for i, bead in enumerate(circle_beads):
            angle = i * (TAU / len(beads))
            bead.move_to(necklace_circle.point_at_angle(angle))
            
        self.play(
            ReplacementTransform(necklace_line, necklace_circle),
            ReplacementTransform(beads, circle_beads),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        # Scissors icon (using two lines to form an X-like shape representing cut)
        scissor_l1 = Line(ORIGIN, UP*0.3).rotate(PI/4)
        scissor_l2 = Line(ORIGIN, UP*0.3).rotate(-PI/4)
        scissors = VGroup(scissor_l1, scissor_l2).set_color(WHITE).set_stroke(width=4)
        
        cut_tracker = ValueTracker(0)
        
        def update_scissors(m):
            angle = cut_tracker.get_value()
            m.move_to(necklace_circle.point_at_angle(angle))
            # Fixed rotation using rotate about center
            m.set_rotation(angle + PI/2)

        # To avoid issues with set_rotation on VGroup (not standard), 
        # we track state or use a method that works for VGroups
        scissors.add_updater(lambda m: m.move_to(necklace_circle.point_at_angle(cut_tracker.get_value())))
        self.add(scissors)
        self.play(Create(scissors))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        # Bar chart components
        bar_red = Rectangle(height=1.5, width=0.5, fill_opacity=0.8, fill_color=RED_COLOR, stroke_width=1)
        bar_blue = Rectangle(height=1.5, width=0.5, fill_opacity=0.8, fill_color=BLUE_COLOR, stroke_width=1)
        
        label_red = Text("R", font_size=18)
        label_blue = Text("B", font_size=18)
        
        # Align labels to bars (once)
        label_red.next_to(bar_red, DOWN, buff=0.1)
        label_blue.next_to(bar_blue, DOWN, buff=0.1)
        
        bar_group = VGroup(bar_red, bar_blue, label_red, label_blue)
        # Issue 36 fix: Changed scale_factor 1.2 to 1.0
        self.place_in_area(bar_group, 'B5', 'D6', scale_factor=1.0)
        
        # Store initial positions for height stretching
        def update_bars(m):
            angle = cut_tracker.get_value()
            # Simulate share values
            val_red = 1.0 + 0.6 * np.sin(angle)
            val_blue = 1.0 + 0.6 * np.cos(angle)
            
            # Use stretch to fit height but ensure bottom stays fixed
            bar_red.stretch_to_fit_height(max(0.1, val_red), about_edge=DOWN)
            bar_blue.stretch_to_fit_height(max(0.1, val_blue), about_edge=DOWN)
            
        bar_red.add_updater(lambda m: update_bars(None))
        
        self.play(FadeIn(bar_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        # Move the cut point around the circle
        self.play(cut_tracker.animate.set_value(TAU * 1.5), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        mapping_text = Text("Positions map to a sphere", font_size=24, color=HIGHLIGHT_COLOR)
        # Issue 37 fix: Changed E2-F5 to E2-F6
        self.place_in_area(mapping_text, 'E2', 'F6', scale_factor=1.0)
        
        self.play(Write(mapping_text))
        self.wait(2)
