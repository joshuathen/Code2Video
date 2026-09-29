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
        self.setup_layout("The Diagnostic Workflow", [
            "Positive tests upload Secret Keys.",
            "Server broadcasts these keys.",
            "Phones reconstruct IDs to check logs.",
            "Matches trigger a user notification.",
            "Privacy is maintained during matching."
        ])
        
        # Assets
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg", color="#FFFFFF")
        alert_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/notification.svg", color=RED)
        shield_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shield.svg", color=PURPLE)
        
        # Elements
        self.place_at_grid(server_icon, 'B3', scale_factor=0.9)
        server_label = Text("Server", font_size=20).next_to(server_icon, UP)
        self.add(server_label)
        
        key_list = VGroup(*[Square(side_length=0.3, color=BLUE).set_fill(BLUE, opacity=0.5) for _ in range(4)])
        key_list.arrange(DOWN, buff=0.1)
        self.place_at_grid(key_list, 'C2', scale_factor=0.8)
        
        log_entry = VGroup(*[Square(side_length=0.3, color=GREEN).set_fill(GREEN, opacity=0.5) for _ in range(4)])
        log_entry.arrange(DOWN, buff=0.1)
        self.place_at_grid(log_entry, 'C5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(server_icon), key_list.animate.move_to(server_icon.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        broadcast = Line(server_icon.get_right(), server_icon.get_right() + RIGHT * 2, color=YELLOW)
        self.play(Create(broadcast))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        self.play(FadeIn(log_entry))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        self.place_at_grid(alert_icon, 'D6', scale_factor=0.7)
        self.play(Flash(alert_icon))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PURPLE)
        self.place_at_grid(shield_icon, 'F3', scale_factor=0.8)
        self.play(FadeIn(shield_icon), FadeOut(broadcast), FadeOut(key_list), FadeOut(log_entry), FadeOut(alert_icon))
        self.wait(2)
