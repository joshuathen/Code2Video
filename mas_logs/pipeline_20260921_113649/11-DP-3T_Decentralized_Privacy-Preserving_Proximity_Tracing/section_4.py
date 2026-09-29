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
        self.setup_layout("Exposure Detection: The Decentralized Match", [
            "Infected users upload Daily Keys.", 
            "Devices download keys to check.", 
            "Matches indicate potential exposure risks."
        ])
        
        # Load Assets
        server = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg", color=BLUE)
        self.place_at_grid(server, 'B2', scale_factor=0.5)
        server_label = Text("Server", font_size=20).next_to(server, DOWN)
        self.add(server_label)
        
        device = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg", color=WHITE)
        self.place_at_grid(device, 'B4', scale_factor=0.5)
        device_label = Text("User Phone", font_size=20).next_to(device, DOWN)
        self.add(device_label)
        
        # === Animation for Lecture Line 1 ===
        # Using SVGMobject requires color overrides sometimes, but here let's add dots for keys
        key_list = VGroup(*[Dot(color=RED) for _ in range(5)]).arrange(DOWN, buff=0.1)
        self.place_in_area(key_list, 'D4', 'F6', scale_factor=0.9)
        self.play(self.lecture[0].animate.set_color(RED), FadeIn(key_list))
        self.play(key_list.animate.move_to(server.get_center()))
        
        # === Animation for Lecture Line 2 ===
        download_key = Dot(color=YELLOW)
        self.place_at_grid(download_key, 'B3', scale_factor=0.6)
        self.play(self.lecture[1].animate.set_color(YELLOW), FadeIn(download_key))
        self.play(download_key.animate.move_to(device.get_center()))
        
        # === Animation for Lecture Line 3 ===
        match_highlight = SurroundingRectangle(device, color=GREEN, buff=0.1)
        self.play(self.lecture[2].animate.set_color(GREEN), Create(match_highlight))
        self.wait(2)
