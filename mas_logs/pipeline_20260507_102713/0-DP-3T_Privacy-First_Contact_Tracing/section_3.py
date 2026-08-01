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
        lecture_lines = [
            'Each phone generates a secret daily key locally.',
            'Every 15 minutes, it derives a new Ephemeral ID.',
            'Hashing the secret key ensures IDs are unlinkable.',
            'Alice broadcasts these rotating nicknames via Bluetooth.',
            'Frequent rotation prevents long-term tracking of individuals.'
        ]
        self.setup_layout("Phase 1: Generating Ephemeral IDs (The Nicknames)", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#3498db"))
        
        phone_body = RoundedRectangle(height=4, width=2.5, corner_radius=0.2, color=GRAY)
        self.place_in_area(phone_body, "B2", "E4")
        
        key_box = Rectangle(height=0.8, width=1.5, color="#3498db", fill_opacity=0.3)
        key_label = Text("Secret Key", font_size=20, color="#3498db")
        key_group = VGroup(key_box, key_label)
        self.place_at_grid(key_group, "C3", scale_factor=0.8)
        
        self.play(Create(phone_body), FadeIn(key_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        clock_face = Circle(radius=0.4, color=WHITE)
        clock_hand = Line(ORIGIN, UP * 0.35, color=WHITE)
        clock = VGroup(clock_face, clock_hand)
        self.place_at_grid(clock, "A4", scale_factor=0.8)
        
        timer_label_base = Text("00:", font_size=24)
        timer_seconds = DecimalNumber(0, num_decimal_places=0, font_size=24, include_sign=False, mob_class=Text)
        timer_group = VGroup(timer_label_base, timer_seconds).arrange(RIGHT, buff=0.1)
        self.place_at_grid(timer_group, "A5", scale_factor=0.8)
        
        self.play(Create(clock), Write(timer_group))
        self.play(
            Rotate(clock_hand, -2*PI, about_point=clock_face.get_center()),
            ChangeDecimalToValue(timer_seconds, 15),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#2ecc71")
        )
        
        hash_box = Rectangle(height=0.8, width=1.2, color="#2ecc71", fill_opacity=0.3)
        hash_label = Text("Hash", font_size=24, color="#2ecc71")
        hash_group = VGroup(hash_box, hash_label)
        self.place_at_grid(hash_group, "D5", scale_factor=0.7)
        
        # Merge animation
        key_copy = key_group.copy()
        time_copy = VGroup(clock.copy(), timer_group.copy())
        
        self.play(FadeIn(hash_group))
        self.play(
            key_copy.animate.move_to(hash_group.get_center()).scale(0.1).set_opacity(0),
            time_copy.animate.move_to(hash_group.get_center()).scale(0.1).set_opacity(0),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#f1c40f")
        )
        
        ephid_label = Text("EphID_1", font_size=32, color="#f1c40f")
        self.place_at_grid(ephid_label, "D6", scale_factor=0.6)
        
        # Output from Hash
        self.play(ReplacementTransform(hash_group.copy(), ephid_label))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#f1c40f")
        )
        
        # Pulse EphID
        self.play(ephid_label.animate.scale(1.2), rate_func=there_and_back)
        
        # Bluetooth waves (arcs)
        waves = VGroup()
        for i in range(3):
            wave = Arc(
                radius=0.4 + i*0.4, 
                start_angle=-PI/3, 
                angle=2*PI/3, 
                color="#f1c40f", 
                stroke_width=2
            )
            wave.move_to(ephid_label.get_center())
            waves.add(wave)
        
        self.play(
            LaggedStart(
                *[
                    VGroup(w).animate(run_time=1.5).scale(2).set_stroke(opacity=0)
                    for w in waves
                ],
                lag_ratio=0.3
            )
        )
        
        self.wait(2)
