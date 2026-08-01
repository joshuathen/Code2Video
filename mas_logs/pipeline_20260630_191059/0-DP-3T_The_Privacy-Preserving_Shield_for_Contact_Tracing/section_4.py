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
        title_text = "The Encounter: Recording the Digital Handshake"
        lecture_lines = [
            "When phones meet, they exchange rotating Ephemeral IDs.",
            "Your phone records the signal strength and ID.",
            "This data is stored locally on your device only.",
            "No location or personal information is ever exchanged.",
            "The central server knows nothing about this meeting."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        alice_color = "#00BFFF"
        bob_color = "#FF69B4"
        diary_color = "#F5DEB3"
        ghost_color = "#A9A9A9"
        server_color = "#808080"
        red_color = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # When phones meet, they exchange rotating Ephemeral IDs.
        self.lecture[0].set_color(alice_color)
        
        alice_icon = Circle(radius=0.4, color=alice_color, fill_opacity=0.3)
        alice_label = Text("Alice", font_size=18, color=alice_color).next_to(alice_icon, DOWN, buff=0.1)
        alice = VGroup(alice_icon, alice_label)
        
        bob_icon = Circle(radius=0.4, color=bob_color, fill_opacity=0.3)
        bob_label = Text("Bob", font_size=18, color=bob_color).next_to(bob_icon, DOWN, buff=0.1)
        bob = VGroup(bob_icon, bob_label)
        
        self.place_at_grid(alice, 'B2')
        self.place_at_grid(bob, 'B5')
        
        self.play(FadeIn(alice), FadeIn(bob))
        self.play(
            alice.animate.move_to(self.grid['B3']),
            bob.animate.move_to(self.grid['B4'])
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Your phone records the signal strength and ID.
        self.lecture[1].set_color(bob_color)
        
        id_a = Text("ID_A1", font_size=14, color=WHITE)
        id_b = Text("ID_B1", font_size=14, color=WHITE)
        
        self.place_at_grid(id_a, 'B3')
        self.place_at_grid(id_b, 'B4')
        
        self.play(
            id_a.animate.move_to(self.grid['B4']),
            id_b.animate.move_to(self.grid['B3']),
            run_time=2
        )
        
        signal_strength = Text("RSSI: -65dBm", font_size=12, color=YELLOW)
        # Position signal_strength in area C3-C4 to balance between devices (Issue 36)
        self.place_in_area(signal_strength, 'C3', 'C4', scale_factor=0.8)
        self.play(Write(signal_strength))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This data is stored locally on your device only.
        self.lecture[2].set_color(diary_color)
        
        # Use SVGMobject for 'Local Diary' icon (Issue 25)
        # Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg
        diary_a = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg", height=0.6).set_color(diary_color)
        label_diary_a = Text("Local Diary", font_size=12, color=diary_color).next_to(diary_a, DOWN, buff=0.1)
        diary_group_a = VGroup(diary_a, label_diary_a)
        
        diary_b = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg", height=0.6).set_color(diary_color)
        label_diary_b = Text("Local Diary", font_size=12, color=diary_color).next_to(diary_b, DOWN, buff=0.1)
        diary_group_b = VGroup(diary_b, label_diary_b)
        
        # Position diary groups to align with phones at D2 and D5 (Issue 34)
        self.place_at_grid(diary_group_a, 'D2')
        self.place_at_grid(diary_group_b, 'D5')
        
        self.play(FadeIn(diary_group_a), FadeIn(diary_group_b))
        
        # Move IDs into diaries
        self.play(
            id_b.animate.move_to(diary_a.get_center()).scale(0.5),
            id_a.animate.move_to(diary_b.get_center()).scale(0.5),
            signal_strength.animate.move_to(diary_a.get_center()).scale(0.5).set_opacity(0)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # No location or personal information is ever exchanged.
        self.lecture[3].set_color(ghost_color)
        
        gps_base = Circle(radius=0.15, color=ghost_color)
        gps_point = Triangle(color=ghost_color).scale(0.15).rotate(180*DEGREES).next_to(gps_base, DOWN, buff=0)
        gps_icon = VGroup(gps_base, gps_point)
        
        id_head = Circle(radius=0.1, color=ghost_color)
        id_body = Rectangle(width=0.3, height=0.2, color=ghost_color).next_to(id_head, DOWN, buff=0.05)
        id_icon = VGroup(id_head, id_body)
        
        cross_line_gps = Line(LEFT, RIGHT, color=red_color).scale(0.2).rotate(45*DEGREES).move_to(gps_icon)
        cross_line_id = Line(LEFT, RIGHT, color=red_color).scale(0.2).rotate(45*DEGREES).move_to(id_icon)
        
        no_gps = VGroup(gps_icon, cross_line_gps)
        no_id = VGroup(id_icon, cross_line_id)
        
        # Position privacy icons to align with devices at E2 and E5 (Issue 35)
        self.place_at_grid(no_gps, 'E2')
        self.place_at_grid(no_id, 'E5')
        
        label_no_gps = Text("No GPS", font_size=12, color=ghost_color).next_to(no_gps, DOWN, buff=0.1)
        label_no_id = Text("No Identity", font_size=12, color=ghost_color).next_to(no_id, DOWN, buff=0.1)
        
        self.play(FadeIn(no_gps), FadeIn(no_id), FadeIn(label_no_gps), FadeIn(label_no_id))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The central server knows nothing about this meeting.
        self.lecture[4].set_color(server_color)
        
        server_box = Rectangle(width=0.8, height=1.0, color=server_color, fill_opacity=0.2)
        l1 = Line(LEFT*0.3, RIGHT*0.3, color=server_color).shift(UP*0.2)
        l2 = Line(LEFT*0.3, RIGHT*0.3, color=server_color)
        l3 = Line(LEFT*0.3, RIGHT*0.3, color=server_color).shift(DOWN*0.2)
        server_label = Text("SERVER", font_size=12, color=server_color).next_to(server_box, UP, buff=0.1)
        server = VGroup(server_box, l1, l2, l3, server_label)
        
        self.place_at_grid(server, 'F6')
        
        connection_line = DashedLine(self.grid['B4'], self.grid['F6'], color=server_color)
        
        cross_mark = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color=red_color),
            Line(UP+RIGHT, DOWN+LEFT, color=red_color)
        ).scale(0.3).move_to(self.grid['E5'])
        
        self.play(FadeIn(server))
        self.play(Create(connection_line))
        self.play(FadeIn(cross_mark))
        self.wait(2)
