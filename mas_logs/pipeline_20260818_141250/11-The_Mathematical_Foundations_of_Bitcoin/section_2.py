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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Proof-of-Work Math: Solving the Puzzle", [
            "Bitcoin requires solving SHA-256 puzzles.", 
            "Miners find a nonce for valid hashes.", 
            "Difficulty ensures a ten-minute block time."
        ])
        
        # Load Assets
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        nonce_label = Text("Nonce:", font_size=32, color=WHITE)
        nonce_val = Integer(0, font_size=32, color="#00FF00")
        comp = computer_icon.copy().scale(0.5)
        nonce_group = VGroup(comp, nonce_label, nonce_val).arrange(RIGHT, buff=0.2)
        
        self.place_at_grid(nonce_group, 'B4', scale_factor=0.8)
        self.add(nonce_group)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        hash_label = Text("Hash:", font_size=32, color=WHITE)
        hash_val = Text("0000a1b2...", font_size=32, color=WHITE)
        hash_group = VGroup(hash_label, hash_val).arrange(RIGHT, buff=0.2)
        
        self.place_at_grid(hash_group, 'C4', scale_factor=0.8)
        self.add(hash_group)
        
        # Simulate incrementing nonce and hash change
        for i in range(10):
            nonce_val.set_value(i * 12345)
            self.wait(0.1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        serv = server_icon.copy().scale(0.5)
        hash_val.set_text("0000c3d4...")
        hash_val.set_color("#FF0000")
        
        diff_label = Text("Difficulty: 10 mins", font_size=32, color=WHITE)
        diff_group = VGroup(serv, diff_label).arrange(RIGHT, buff=0.2)
        
        self.place_at_grid(diff_group, 'D4', scale_factor=0.8)
        self.add(diff_group)
        self.wait(2)
