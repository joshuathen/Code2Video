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
        # Initial Setup
        title = "Step 1: Generating Daily Secrets and Ephemeral IDs"
        lines = [
            "Alice's phone generates a random Daily Secret Key.",
            "This key creates changing Ephemeral IDs every fifteen minutes.",
            "To others, these IDs look like random, unrelated strings.",
            "Only Alice’s phone knows the secret behind these IDs.",
            "This rotation prevents tracking Alice's movements over time."
        ]
        self.setup_layout(title, lines)

        # Assets & Colors
        PHONE_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg"
        SK_COLOR = "#F1C40F"
        EPH_COLOR = "#3498DB"
        OBSERVER_COLOR = "#95A5A6"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(SK_COLOR)
        
        # Phone
        phone = SVGMobject(PHONE_ASSET)
        self.place_at_grid(phone, "B2", scale_factor=0.8)
        
        # SK_t Label
        sk_label = Text("SK_t", color=SK_COLOR, font_size=20)
        self.place_at_grid(sk_label, "B2", scale_factor=0.8)
        sk_label.shift(DOWN * 0.1) # Center inside phone
        
        # Glowing Key (represented by a circle behind label)
        glow = Arc(radius=0.3, angle=TAU, color=SK_COLOR).move_to(sk_label)
        glow.add_updater(lambda m, dt: m.set_opacity(abs(np.sin(self.time * 2))))
        
        self.play(FadeIn(phone), Write(sk_label), FadeIn(glow))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(EPH_COLOR)

        # Derivation process
        hash_box = Rectangle(height=0.6, width=1.2, color=WHITE).set_fill(GREY_E, opacity=1)
        hash_text = Text("Hash", font_size=18).move_to(hash_box)
        hash_group = VGroup(hash_box, hash_text)
        self.place_at_grid(hash_group, "B4", scale_factor=1.0)

        # Arrows and EphIDs
        eph_labels = VGroup(*[
            Text(f"EphID_{i+1}", color=EPH_COLOR, font_size=16)
            for i in range(5)
        ])
        
        grid_positions = ["D1", "D2", "D3", "D4", "D5"]
        for i, pos in enumerate(grid_positions):
            self.place_at_grid(eph_labels[i], pos, scale_factor=1.0)

        arrow_to_hash = Arrow(phone.get_right(), hash_group.get_left(), buff=0.1, color=WHITE)
        
        self.play(Create(arrow_to_hash), FadeIn(hash_group))
        
        # Animate EphIDs popping out
        spawn_anims = []
        for eph in eph_labels:
            spawn_anims.append(ReplacementTransform(hash_group.copy(), eph))
        
        self.play(AnimationGroup(*spawn_anims, lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(OBSERVER_COLOR)

        # Observer Icon (Generic Circle with label)
        observer = Circle(radius=0.4, color=OBSERVER_COLOR).set_fill(OBSERVER_COLOR, opacity=0.3)
        obs_text = Text("Observer", font_size=14, color=OBSERVER_COLOR).next_to(observer, DOWN, buff=0.1)
        observer_group = VGroup(observer, obs_text)
        self.place_at_grid(observer_group, "B6", scale_factor=1.0)

        q_marks = Text("???", color=OBSERVER_COLOR, font_size=24).next_to(observer, UP, buff=0.2)

        # Observer looks at EphIDs
        eye_lines = VGroup(*[
            Line(observer.get_bottom(), eph.get_top(), color=OBSERVER_COLOR, stroke_width=1, stroke_opacity=0.5)
            for eph in eph_labels
        ])

        self.play(FadeIn(observer_group), Write(q_marks))
        self.play(Create(eye_lines))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(SK_COLOR)

        # Dashed connection back to SK_t
        connections = VGroup(*[
            DashedLine(sk_label.get_center(), eph.get_center(), color=SK_COLOR, stroke_width=2)
            for eph in eph_labels
        ])

        # Remove observer links first to clear view
        self.play(FadeOut(eye_lines), FadeOut(q_marks), FadeOut(observer_group))
        self.play(Create(connections))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(EPH_COLOR)

        # Move phone and highlight current ID
        # Positions B2, B3, B4, B5
        move_grid = ["B1", "B2", "B3", "B4", "B5"]
        
        # Cleanup
        self.play(FadeOut(connections), FadeOut(hash_group), FadeOut(arrow_to_hash))
        
        # Phone Group
        phone_group = VGroup(phone, sk_label, glow)
        
        for i, pos in enumerate(move_grid):
            target_center = self.grid[pos]
            self.play(
                phone_group.animate.move_to(target_center),
                eph_labels[i].animate.set_color(YELLOW).scale(1.2),
                run_time=0.8
            )
            self.play(eph_labels[i].animate.set_color(EPH_COLOR).scale(1/1.2), run_time=0.2)

        self.wait(2)
