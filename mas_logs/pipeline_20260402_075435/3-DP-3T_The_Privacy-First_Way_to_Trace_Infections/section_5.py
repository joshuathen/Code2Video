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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Bob’s phone downloads new diagnosis keys daily.",
            "It locally regenerates the IDs from those seeds.",
            "The phone checks for matches in its local log.",
            "A match triggers a private exposure alert.",
            "All matching happens locally, never on the server."
        ]
        self.setup_layout("Phase 3: Local Matching & Alert (1:00)", lecture_lines)

        # Assets setup
        # Bob's Phone [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg]
        phone_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg"
        phone = SVGMobject(phone_path, color="#E67E22")
        self.place_in_area(phone, "B1", "E2", scale_factor=1.2)
        
        # Cloud
        cloud = VGroup(
            Circle(radius=0.3, color=WHITE, fill_opacity=0.8),
            Circle(radius=0.2, color=WHITE, fill_opacity=0.8).shift(LEFT*0.3),
            Circle(radius=0.2, color=WHITE, fill_opacity=0.8).shift(RIGHT*0.3),
            Rectangle(width=0.6, height=0.3, color=WHITE, fill_opacity=0.8).shift(DOWN*0.1)
        )
        self.place_at_grid(cloud, "A5", scale_factor=0.8)
        cloud_label = Text("Cloud Server", font_size=16, color=WHITE).next_to(cloud, UP, buff=0.1)

        # Bob Icon
        bob_head = Circle(radius=0.15, color="#E67E22", fill_opacity=1)
        bob_body = Triangle(color="#E67E22", fill_opacity=1).scale(0.3).next_to(bob_head, DOWN, buff=0)
        bob_icon = VGroup(bob_head, bob_body)
        self.place_at_grid(bob_icon, "F2")
        bob_label = Text("Bob", font_size=18, color="#E67E22").next_to(bob_icon, DOWN, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#E67E22"))
        self.play(Create(phone), Create(cloud), Create(cloud_label), Create(bob_icon), Create(bob_label))
        
        secret_seed = Text("Secret Seed", font_size=14, color="#E67E22", weight=BOLD)
        # Initialize at cloud
        self.place_at_grid(secret_seed, "A5")
        
        self.play(FadeIn(secret_seed))
        # Move seed to phone center
        phone_center = (self.grid["B1"] + self.grid["E2"]) / 2
        self.play(secret_seed.animate.move_to(phone_center), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#95A5A6"))
        
        # Blender graphic - Issue 40: place at A2
        blender = RegularPolygon(n=3, color="#95A5A6", fill_opacity=0.5).rotate(PI)
        blender_text = Text("Blender", font_size=14, color=WHITE).move_to(blender.get_center())
        blender_group = VGroup(blender, blender_text)
        self.place_at_grid(blender_group, "A2", scale_factor=0.8)
        
        self.play(FadeIn(blender_group))
        
        gen_ids = VGroup(
            Text("ID-X", font_size=14),
            Text("ID-Alpha", font_size=14),
            Text("ID-Z", font_size=14)
        ).arrange(DOWN, buff=0.2)
        self.place_at_grid(gen_ids, "D2", scale_factor=1.0) # Inside phone screen
        
        self.play(
            FadeOut(secret_seed),
            LaggedStart(*[FadeIn(id_obj) for id_obj in gen_ids], lag_ratio=0.3)
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#F1C40F"))
        
        # Issue 42: place local_log at D4
        local_log_box = RoundedRectangle(height=2, width=1.5, color=WHITE, stroke_width=2)
        local_log_title = Text("Local Log", font_size=14, color=WHITE)
        local_log_entries = VGroup(
            Text("ID-B", font_size=14),
            Text("ID-Alpha", font_size=14),
            Text("ID-K", font_size=14)
        ).arrange(DOWN, buff=0.2)
        local_log_inner = VGroup(local_log_title, local_log_entries).arrange(DOWN, buff=0.1)
        
        self.place_at_grid(local_log_inner, "D4", scale_factor=1.0)
        local_log_box.move_to(local_log_inner.get_center())
        local_log = VGroup(local_log_box, local_log_inner)
        
        self.play(Create(local_log))
        
        # Highlight match
        match_highlight_1 = SurroundingRectangle(gen_ids[1], color="#F1C40F", buff=0.05)
        match_highlight_2 = SurroundingRectangle(local_log_entries[1], color="#F1C40F", buff=0.05)
        
        self.play(Create(match_highlight_1), Create(match_highlight_2))
        self.play(gen_ids[1].animate.set_color("#F1C40F"), local_log_entries[1].animate.set_color("#F1C40F"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#E74C3C"))
        
        # Issue 41: alert_group at B2
        alert_banner = Rectangle(height=0.4, width=1.4, fill_color="#E74C3C", fill_opacity=1, stroke_width=0)
        alert_text = Text("EXPOSURE ALERT", font_size=12, color=WHITE, weight=BOLD)
        alert_group = VGroup(alert_banner, alert_text)
        self.place_at_grid(alert_group, "B2") 
        
        self.play(FadeIn(alert_group, scale=1.2))
        # Bob icon pulses
        self.play(bob_icon.animate.scale(1.2).set_color("#E74C3C"), run_time=0.5)
        self.play(bob_icon.animate.scale(1/1.2), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        # Cross out the cloud connection to emphasize local
        cross_x = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color=RED),
            Line(UP+RIGHT, DOWN+LEFT, color=RED)
        ).scale(0.5)
        self.place_at_grid(cross_x, "B4")
        
        self.play(Create(cross_x))
        self.play(Indicate(phone, color="#FFFFFF"))
        self.wait(2)
