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

class Section7Scene(TeachingScene):
    def construct(self):
        title = "Summary: Why it’s Secure"
        lecture_lines = [
            "DP-3T never uses or records your GPS location.",
            "Your identity remains anonymous to the central server.",
            "All sensitive matching happens locally on your phone."
        ]
        self.setup_layout(title, lecture_lines)

        # Define Colors
        COLOR_1 = "#ADD8E6"  # Light Blue
        COLOR_2 = "#90EE90"  # Light Green
        COLOR_3 = "#FFFFE0"  # Light Yellow
        SERVER_COLOR = "#808080"
        KEY_COLOR = "#FFFFFF"
        PHONE_COLOR = "#C0C0C0"
        CHECK_COLOR = "#00FF00"
        
        # Asset paths
        PHONE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        SERVER_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/server.svg"

        # === Animation for Lecture Line 1 ===
        # Map Pin with X (No Location)
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        pin_circle = Circle(radius=0.3, color=COLOR_1, fill_opacity=0.3)
        pin_point = Triangle(color=COLOR_1, fill_opacity=0.3).scale(0.3).rotate(180*DEGREES).next_to(pin_circle, DOWN, buff=0)
        map_pin = VGroup(pin_circle, pin_point)
        
        x_mark = VGroup(
            Line(LEFT, RIGHT).rotate(45*DEGREES),
            Line(LEFT, RIGHT).rotate(-45*DEGREES)
        ).set_color(RED).scale(0.4).move_to(map_pin)
        
        no_location_icon = VGroup(map_pin, x_mark)
        self.place_at_grid(no_location_icon, "B2", scale_factor=0.8)
        
        loc_label = Text("No GPS", font_size=18, color=COLOR_1).next_to(no_location_icon, DOWN, buff=0.2)
        
        self.play(FadeIn(no_location_icon), Write(loc_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Pseudonymous Mask and Server with Keys
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_2)
        )
        
        # Mask Icon
        mask_base = RoundedRectangle(height=0.4, width=0.8, corner_radius=0.1, color=COLOR_2, fill_opacity=0.3)
        eye_l = Circle(radius=0.05, color=COLOR_2, fill_opacity=1).move_to(mask_base.get_center() + LEFT*0.2)
        eye_r = Circle(radius=0.05, color=COLOR_2, fill_opacity=1).move_to(mask_base.get_center() + RIGHT*0.2)
        mask_icon = VGroup(mask_base, eye_l, eye_r)
        self.place_at_grid(mask_icon, "D2", scale_factor=1.0)
        mask_label = Text("Pseudonym", font_size=18, color=COLOR_2).next_to(mask_icon, DOWN, buff=0.2)
        
        # Server Icon [Asset]
        server_icon = SVGMobject(SERVER_ASSET).set_color(SERVER_COLOR)
        self.place_at_grid(server_icon, "D5", scale_factor=1.0)
        server_label = Text("Server", font_size=18, color=SERVER_COLOR).next_to(server_icon, DOWN, buff=0.2)
        
        # Keys
        key_head = Circle(radius=0.1, color=KEY_COLOR, fill_opacity=0.5)
        key_stem = Line(ORIGIN, DOWN*0.2, color=KEY_COLOR).next_to(key_head, DOWN, buff=0)
        key_icon = VGroup(key_head, key_stem)
        self.place_at_grid(key_icon, "D6", scale_factor=0.6)
        key_label = Text("Keys Only", font_size=16, color=KEY_COLOR).next_to(key_icon, DOWN, buff=0.2)
        
        # Separation wall
        wall = DashedLine(UP, DOWN, color=WHITE).scale(1.5)
        self.place_at_grid(wall, "D4")

        self.play(FadeIn(mask_icon), Write(mask_label))
        self.play(FadeIn(server_icon), Write(server_label), FadeIn(wall))
        self.play(FadeIn(key_icon), Write(key_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Local Matching (Phone with gears) and Final Privacy check
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_3)
        )
        
        # Phone with gears [Asset]
        phone_matching = SVGMobject(PHONE_ASSET).set_color(COLOR_3)
        gear1 = Star(n=8, color=COLOR_3).scale(0.1).move_to(phone_matching.get_center() + UP*0.1)
        gear2 = Star(n=8, color=COLOR_3).scale(0.1).move_to(phone_matching.get_center() + DOWN*0.1)
        local_matching_icon = VGroup(phone_matching, gear1, gear2)
        self.place_at_grid(local_matching_icon, "F1", scale_factor=0.8)
        local_label = Text("Local Match", font_size=18, color=COLOR_3).next_to(local_matching_icon, DOWN, buff=0.2)
        
        self.play(FadeIn(local_matching_icon), Write(local_label))
        
        # Final View: Two Phones and Checkmark [Assets]
        phone1 = SVGMobject(PHONE_ASSET).set_color(PHONE_COLOR)
        phone2 = SVGMobject(PHONE_ASSET).set_color(PHONE_COLOR)
        self.place_at_grid(phone1, "E1", scale_factor=0.8)
        self.place_at_grid(phone2, "E3", scale_factor=0.8)
        
        checkmark = VGroup(
            Line(ORIGIN, RIGHT*0.2 + DOWN*0.2),
            Line(RIGHT*0.2 + DOWN*0.2, RIGHT*0.5 + UP*0.4)
        ).set_color(CHECK_COLOR).set_stroke(width=6)
        self.place_at_grid(checkmark, "E2", scale_factor=0.6)
        
        privacy_text = Text("Privacy Preserved", font_size=20, color=CHECK_COLOR)
        self.place_in_area(privacy_text, "F2", "F3", scale_factor=0.8)

        self.play(
            FadeIn(phone1),
            FadeIn(phone2),
            Create(checkmark)
        )
        self.play(Write(privacy_text))
        
        self.wait(2)
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
