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
        # Fetching data from shared state
        title_str = "Step 1: The Oracle (Phase Inversion)"
        lecture_lines = [
            "We start with four bars, labeling one as our target.",
            "The Oracle flips the target amplitude below the axis.",
            "This pulsing yellow border highlights the marked state.",
            "This process is known as a phase inversion.",
            "The target's probability magnitude remains the same."
        ]
        
        self.setup_layout(title_str, lecture_lines)
        
        # Colors
        COLOR_TARGET = "#FFFF00"  # Yellow
        COLOR_BAR = "#87CEEB"     # Sky Blue
        COLOR_TEXT = "#FFFFFF"    # White
        COLOR_PROB = "#ADD8E6"    # Light Blue
        COLOR_AXIS = "#666666"    # Dark Gray
        
        # === Animation for Lecture Line 1 ===
        # Show 4 bars, label '10' as 'Target' (#FFFF00)
        self.lecture[0].set_color(COLOR_TARGET)
        
        # Horizontal Axis at Row D
        axis_line = Line(self.grid['D1'], self.grid['D6'], color=COLOR_AXIS)
        self.add(axis_line)
        
        # Bars: Rectangle height 1.0 fits between two rows.
        # Positive bars sit in Row C-D.
        bar00 = Rectangle(width=0.6, height=1.0, fill_opacity=0.8, fill_color=COLOR_BAR, stroke_color=WHITE, stroke_width=1)
        bar01 = Rectangle(width=0.6, height=1.0, fill_opacity=0.8, fill_color=COLOR_BAR, stroke_color=WHITE, stroke_width=1)
        bar10 = Rectangle(width=0.6, height=1.0, fill_opacity=0.8, fill_color=COLOR_TARGET, stroke_color=WHITE, stroke_width=1)
        bar11 = Rectangle(width=0.6, height=1.0, fill_opacity=0.8, fill_color=COLOR_BAR, stroke_color=WHITE, stroke_width=1)
        
        self.place_in_area(bar00, 'C2', 'D2')
        self.place_in_area(bar01, 'C3', 'D3')
        self.place_in_area(bar10, 'C4', 'D4')
        self.place_in_area(bar11, 'C5', 'D5')
        
        # Index labels at Row B (above the bars)
        label00 = Text("00", font_size=20, color=COLOR_TEXT)
        label01 = Text("01", font_size=20, color=COLOR_TEXT)
        label10 = Text("10", font_size=20, color=COLOR_TARGET)
        label11 = Text("11", font_size=20, color=COLOR_TEXT)
        
        self.place_at_grid(label00, 'B2')
        self.place_at_grid(label01, 'B3')
        self.place_at_grid(label10, 'B4')
        self.place_at_grid(label11, 'B5')
        
        # 'Target' label - Initial placement at A4 (scaled down for buffer L006)
        target_marker = Text("Target", font_size=22, color=COLOR_TARGET)
        self.place_at_grid(target_marker, 'A4', scale_factor=0.6) 
        
        self.play(
            FadeIn(VGroup(bar00, bar01, bar10, bar11)),
            Write(VGroup(label00, label01, label10, label11)),
            FadeIn(target_marker)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # The 'Target' bar flips below the horizontal x-axis
        self.lecture[1].set_color(COLOR_TEXT)
        
        # Calculate target position for the flipped bar (D4-E4)
        target_rect_ghost = Rectangle(width=0.6, height=1.0)
        self.place_in_area(target_rect_ghost, 'D4', 'E4')
        target_pos = target_rect_ghost.get_center()
        
        self.play(
            bar10.animate.move_to(target_pos),
            # Fix Issue 31: Move target_marker to E3 to avoid overlap and crowding
            target_marker.animate.move_to(self.grid['E3']).scale(1.33) # 0.6 * 1.33 approx 0.8
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Pulsing yellow border highlights the marked state
        self.lecture[2].set_color(COLOR_TARGET)
        
        pulse_box = SurroundingRectangle(bar10, color=COLOR_TARGET, buff=0.1)
        
        pulse_tracker = ValueTracker(0)
        def pulse_updater(m):
            m.set_stroke(width=1 + 3 * np.abs(np.sin(pulse_tracker.get_value() * PI)))
            
        pulse_box.add_updater(pulse_updater)
        
        self.add(pulse_box)
        self.play(
            FadeIn(pulse_box),
            pulse_tracker.animate(run_time=2, rate_func=linear).set_value(4)
        )
        pulse_box.add_updater(lambda m, dt: pulse_tracker.increment_value(dt))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # 'Phase Inversion' label (#FFFFFF)
        self.lecture[3].set_color(COLOR_TEXT)
        
        phase_label = Text("Phase Inversion", font_size=24, color=COLOR_TEXT)
        # Fix Issue 30: place at E6 with scale 0.8 to avoid overlap with target_rect
        self.place_at_grid(phase_label, 'E6', scale_factor=0.8)
        
        self.play(Write(phase_label))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # Probability text (#ADD8E6)
        self.lecture[4].set_color(COLOR_PROB)
        
        prob_text = Text("Probability |a|^2\nremains unchanged", font_size=20, color=COLOR_PROB)
        # Fix Issue 32: place in area F2-F6 with scale 0.6 to prevent cramping
        self.place_in_area(prob_text, 'F2', 'F6', scale_factor=0.6)
        
        self.play(FadeIn(prob_text))
        self.wait(2)
