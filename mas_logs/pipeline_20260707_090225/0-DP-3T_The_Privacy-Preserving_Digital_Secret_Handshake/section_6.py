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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Phase 4: The Private Matching (Risk Calculation)", 
            [
                "Bob's phone periodically downloads the new public sick seeds.",
                "His phone locally generates IDs from these seeds.",
                "It compares these generated IDs to his visitor log.",
                "A match indicates Bob was near a sick person.",
                "Bob receives a private notification to get tested."
            ]
        )
        
        # Define Colors
        BLUE_SEED = "#5DADE2"
        YELLOW_HASH = "#F4D336"
        RED_MATCH = "#EC7063"

        # Static Elements setup
        phone_body = RoundedRectangle(height=5, width=3, corner_radius=0.3, color=WHITE)
        self.place_in_area(phone_body, "C1", "F6")
        
        # Cloud Icon (Simplified)
        cloud_circles = VGroup(
            Circle(radius=0.4, fill_opacity=1, color=GREY_B),
            Circle(radius=0.3, fill_opacity=1, color=GREY_B).shift(LEFT*0.4),
            Circle(radius=0.3, fill_opacity=1, color=GREY_B).shift(RIGHT*0.4)
        )
        cloud_base = Rectangle(width=0.8, height=0.4, fill_opacity=1, color=GREY_B).shift(DOWN*0.1)
        cloud = VGroup(cloud_circles, cloud_base)
        self.place_at_grid(cloud, "B5", scale_factor=0.6)
        
        self.add(phone_body, cloud)

        # === Animation for Lecture Line 1 ===
        # Bob's phone periodically downloads the new public sick seeds.
        self.play(self.lecture[0].animate.set_color(BLUE_SEED))
        
        seeds = VGroup(*[
            Circle(radius=0.1, color=BLUE_SEED, fill_opacity=1) 
            for _ in range(3)
        ]).arrange(RIGHT, buff=0.1)
        # Fix Issue 45: Defined scale factor 0.6
        self.place_at_grid(seeds, "B5", scale_factor=0.6)
        
        # Animate seeds moving from cloud into phone
        self.play(
            seeds.animate.move_to(self.grid["D2"]),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # His phone locally generates IDs from these seeds.
        self.play(self.lecture[1].animate.set_color(YELLOW_HASH))
        
        # Hash Box
        hash_box = Rectangle(height=0.8, width=1.4, color=YELLOW_HASH, fill_opacity=0.2)
        hash_label = Text("Hash Box", font_size=18, color=YELLOW_HASH)
        hash_group = VGroup(hash_box, hash_label)
        self.place_at_grid(hash_group, "D2", scale_factor=0.8)
        self.play(FadeIn(hash_group))

        # Generated ID
        gen_id = Text("XJ-9", font_size=14, color=YELLOW_HASH)
        # Fix Issue 43: Move gen_id to D3 to avoid overlap with hash_group at D2
        self.place_at_grid(gen_id, "D3", scale_factor=0.8)
        
        self.play(
            seeds.animate.scale(0.1).set_opacity(0),
            FadeIn(gen_id),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # It compares these generated IDs to his visitor log.
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Visitor Log
        log_title = Text("Visitor Log", font_size=16, color=WHITE)
        log_entries = VGroup(
            Text("KY-2", font_size=14),
            Text("XJ-9", font_size=14),
            Text("TR-7", font_size=14)
        ).arrange(DOWN, buff=0.2)
        visitor_log = VGroup(log_title, log_entries).arrange(DOWN, buff=0.2)
        log_bg = Rectangle(height=2.2, width=1.6, color=WHITE, fill_opacity=0.05)
        log_ui = VGroup(log_bg, visitor_log)
        self.place_at_grid(log_ui, "D5", scale_factor=0.8)
        
        self.play(FadeIn(log_ui))

        # ID moves next to log
        self.play(
            gen_id.animate.move_to(self.grid["D4"]),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # A match indicates Bob was near a sick person.
        self.play(self.lecture[3].animate.set_color(RED_MATCH))
        
        # Find matching ID in log
        match_in_log = log_entries[1] # XJ-9
        
        self.play(
            match_in_log.animate.set_color(RED_MATCH),
            gen_id.animate.set_color(RED_MATCH)
        )
        
        self.play(
            Flash(match_in_log, color=RED_MATCH, flash_radius=0.3),
            match_in_log.animate.scale(1.2),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Bob receives a private notification to get tested.
        self.play(self.lecture[4].animate.set_color(RED_MATCH))

        # Notification Box
        notif_bg = RoundedRectangle(height=0.8, width=2.4, color=RED_MATCH, fill_opacity=0.9, corner_radius=0.1)
        notif_text = Text("Potential Exposure", font_size=16, color=WHITE)
        notification = VGroup(notif_bg, notif_text)
        # Fix Issue 44: Place notification in area F2-F4 to avoid cut-off
        self.place_in_area(notification, "F2", "F4", scale_factor=0.8)
        
        self.play(
            FadeIn(notification, shift=UP),
            run_time=1
        )
        
        self.wait(3)
