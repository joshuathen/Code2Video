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
        # Data from storyboard
        title_text = "Step 1: Local Key Generation (The Secret Seed)"
        lecture_lines = [
            "Every day, Alice's phone generates a new Secret Key.",
            "This key produces temporary IDs every fifteen minutes.",
            "Alice broadcasts these rotating IDs via Bluetooth signals."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        SK_COLOR = "#FFD700"
        EPHID_COLOR = "#00FFFF"
        TIMELINE_COLOR = "#FFFFFF"
        PHONE_COLOR = "#808080"
        WAVE_COLOR = "#0000FF"

        # === Animation for Lecture Line 1 ===
        # Every day, Alice's phone generates a new Secret Key.
        self.lecture[0].set_color(SK_COLOR)
        
        # Phone object
        phone_body = RoundedRectangle(height=3, width=1.8, corner_radius=0.2, color=PHONE_COLOR, fill_opacity=0.2)
        phone_screen = Rectangle(height=2.4, width=1.6, color=WHITE, fill_opacity=0.1)
        phone = VGroup(phone_body, phone_screen)
        # Fix Issue 28: self.place_in_area(phone, 'B3', 'D4', scale_factor=0.8)
        self.place_in_area(phone, 'B3', 'D4', scale_factor=0.8)
        
        # SK_t label/icon
        sk_label = MathTex(r"SK_t", color=SK_COLOR, font_size=40)
        # Simple key-like icon
        key_circ = Circle(radius=0.15, color=SK_COLOR)
        key_line = Line(ORIGIN, DOWN*0.3, color=SK_COLOR)
        key_teeth = VGroup(
            Line(ORIGIN, RIGHT*0.1, color=SK_COLOR),
            Line(ORIGIN, RIGHT*0.1, color=SK_COLOR).shift(DOWN*0.1)
        ).shift(DOWN*0.2)
        key_icon = VGroup(key_circ, key_line, key_teeth).next_to(sk_label, UP, buff=0.1)
        sk_group = VGroup(key_icon, sk_label).move_to(phone_screen.get_center())

        self.play(Create(phone), FadeIn(sk_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This key produces temporary IDs every fifteen minutes.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(EPHID_COLOR)
        
        # Timeline
        timeline = NumberLine(
            x_range=[0, 60, 15],
            length=4,
            color=TIMELINE_COLOR,
            include_numbers=True,
            font_size=20,
            label_direction=DOWN
        )
        timeline_label = Text("Time (min)", font_size=16, color=TIMELINE_COLOR).next_to(timeline, RIGHT, buff=0.2)
        timeline_group = VGroup(timeline, timeline_label)
        # Fix Issue 29: self.place_in_area(timeline_group, 'E1', 'F6', scale_factor=1.0)
        self.place_in_area(timeline_group, 'E1', 'F6', scale_factor=1.0)
        
        self.play(Create(timeline_group))
        
        # Sequential Generation
        eph_labels = []
        for i, t_val in enumerate([0, 15, 30, 45]):
            eph_id = MathTex(f"EphID_{i+1}", color=EPHID_COLOR, font_size=24)
            eph_id.move_to(sk_group.get_center())
            
            target_pos = timeline.n2p(t_val) + UP * 0.4
            
            self.play(
                eph_id.animate.move_to(target_pos),
                run_time=0.8
            )
            eph_labels.append(eph_id)
            self.wait(0.2)

        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Alice broadcasts these rotating IDs via Bluetooth signals.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WAVE_COLOR)
        
        # Active EphID near phone
        active_eph = MathTex(r"EphID_n", color=EPHID_COLOR, font_size=32)
        # Fix Issue 30: self.place_at_grid(active_eph, 'B5', scale_factor=0.9)
        self.place_at_grid(active_eph, 'B5', scale_factor=0.9)
        
        # Bluetooth waves pulsing from phone
        def create_wave(radius, opacity):
            return Arc(
                radius=radius, 
                angle=PI/2, 
                start_angle=-PI/4, 
                color=WAVE_COLOR, 
                stroke_width=4
            ).set_stroke(opacity=opacity).move_to(phone.get_right(), aligned_edge=LEFT)

        wave1 = create_wave(0.2, 1)
        
        self.play(FadeIn(active_eph), FadeIn(wave1))
        
        # Pulse animation using a persistent updater for efficiency
        wave_scale = ValueTracker(1)
        wave1.add_updater(lambda m: m.scale(wave_scale.get_value() / m.get_width() * 0.4 if m.get_width() > 0 else 1).set_stroke(opacity=1 - (wave_scale.get_value()-1)/3))
        # Note: The above updater is a bit complex for a simple pulse. Let's stick to a simple loop for now as per budget, 
        # but the prompt prefers persistent mobjects + add_updater.
        # Let's use a simpler updater or just the animation loop since it's only 3 pulses.
        
        for _ in range(3):
            # Create a new wave for each pulse to avoid scale issues with persistent updater
            w = create_wave(0.1, 1)
            self.add(w)
            self.play(
                w.animate.scale(5).set_stroke(opacity=0),
                run_time=1.5,
                rate_func=linear
            )
            self.remove(w)

        self.wait(2)
