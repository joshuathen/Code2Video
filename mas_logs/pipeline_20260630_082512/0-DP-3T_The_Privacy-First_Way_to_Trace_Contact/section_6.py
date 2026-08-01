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
        # Setup layout
        lecture_lines = [
            "DP-3T ensures identity and location remain private.",
            "Matching happens on your device, not a server.",
            "Technology protecting health and privacy simultaneously."
        ]
        self.setup_layout("Conclusion: Security & Privacy Recap", lecture_lines)

        # Colors
        IDENTITY_COLOR = "#00FFFF"
        LOCATION_COLOR = "#FFFF00"
        CHECK_COLOR = "#00FF00"
        HEART_COLOR = "#FF69B4"
        SHIELD_COLOR = "#4682B4"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        identity_label = Text("Identity", font_size=24, color=IDENTITY_COLOR)
        location_label = Text("Location", font_size=24, color=LOCATION_COLOR)
        
        # Use Text instead of MathTex to avoid external 'latex' dependency
        check1 = Text("✔", color=CHECK_COLOR)
        check2 = Text("✔", color=CHECK_COLOR)
        
        self.place_at_grid(identity_label, "B2", scale_factor=0.8)
        self.place_at_grid(check1, "B3", scale_factor=1.2)
        self.place_at_grid(location_label, "D2", scale_factor=0.8)
        self.place_at_grid(check2, "D3", scale_factor=1.2)
        
        self.play(Write(identity_label), Write(location_label))
        self.play(Create(check1), Create(check2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Cleanup Line 1 visuals
        self.play(FadeOut(identity_label), FadeOut(location_label), FadeOut(check1), FadeOut(check2))
        
        # Phone Asset and Blank Server
        phone_asset_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        phone = SVGMobject(phone_asset_path, color=WHITE)
        self.place_at_grid(phone, "C2", scale_factor=0.8)
        
        server = RoundedRectangle(corner_radius=0.1, height=1.5, width=1.0, color=GREY_A, fill_opacity=0.2)
        server_label = Text("Server", font_size=16, color=GREY_A).next_to(server, UP, buff=0.1)
        server_group = VGroup(server, server_label)
        self.place_at_grid(server_group, "C5", scale_factor=1.0)
        
        # Processing animation: dots inside/around phone
        dots = VGroup(*[Dot(radius=0.05, color=BLUE) for _ in range(3)])
        dots.arrange(RIGHT, buff=0.1)
        dots.move_to(phone.get_center())
        
        self.play(FadeIn(phone), FadeIn(server_group))
        
        # Simulate processing with a pulse and dots
        self.play(
            phone.animate.scale(1.1),
            dots.animate.set_opacity(1),
            run_time=0.5
        )
        self.play(
            phone.animate.scale(0.909),
            dots.animate.set_opacity(0),
            run_time=0.5
        )
        
        # No lines to server to show it's blind
        cross = Cross(server, stroke_width=4, color=RED)
        self.play(Create(cross))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Cleanup Line 2 visuals
        self.play(FadeOut(phone), FadeOut(server_group), FadeOut(cross), FadeOut(dots))
        
        # Heart (Use Text instead of MathTex to avoid 'latex' dependency)
        heart = Text("♥", color=HEART_COLOR)
        # Using a simple polygon for a shield shape
        shield = Polygon(
            [-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.2, 0], [0, -0.6, 0], [-0.5, -0.2, 0],
            color=SHIELD_COLOR, fill_opacity=0.8
        )
        
        self.place_at_grid(heart, "B4", scale_factor=1.5)
        self.place_at_grid(shield, "D4", scale_factor=0.8)
        
        self.play(DrawBorderThenFill(heart), DrawBorderThenFill(shield))
        self.wait(0.5)
        
        # Merge at C4
        target_pos = self.grid["C4"]
        self.play(
            heart.animate.move_to(target_pos).scale(0.7),
            shield.animate.move_to(target_pos).scale(1.2),
            run_time=1.5
        )
        
        # Final combined look
        combined_icon = VGroup(shield, heart)
        self.play(combined_icon.animate.scale(1.1).set_opacity(1))
        
        self.wait(2)
