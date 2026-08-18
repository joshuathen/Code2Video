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
        self.setup_layout("Proof of Work: The Lottery of Consensus", [
            "Mining is a race to solve puzzles.",
            "Miners find a nonce for specific hashes.",
            "This secures the network through computational energy."
        ])
        
        # Assets
        miner_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg"
        server_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg"
        
        miners = VGroup(*[SVGMobject(miner_svg, color=WHITE) for _ in range(4)])
        self.place_at_grid(miners[0], 'B1', scale_factor=0.8)
        self.place_at_grid(miners[1], 'B3', scale_factor=0.8)
        self.place_at_grid(miners[2], 'E1', scale_factor=0.8)
        self.place_at_grid(miners[3], 'E3', scale_factor=0.8)

        nonce_labels = VGroup(*[Text("?", font_size=20) for _ in range(4)])
        for i, pos in enumerate(['B1', 'B3', 'E1', 'E3']):
            self.place_at_grid(nonce_labels[i], pos, scale_factor=0.7, offset=UP*0.5)

        mining_simulation_group = VGroup(miners, nonce_labels)
        self.place_in_area(mining_simulation_group, 'A2', 'F5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(miners))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(
            *[Indicate(m) for m in miners],
            *[Write(t) for t in nonce_labels],
            run_time=2
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        winner = SVGMobject(server_svg, color="#00FF00")
        self.place_at_grid(winner, 'E1', scale_factor=0.8)
        
        self.play(
            FadeOut(miners[2]),
            FadeIn(winner),
            nonce_labels[2].animate.become(Text("Nonce Found!", font_size=18, color="#00FF00").next_to(winner, UP, buff=0.1)),
            run_time=2
        )
        self.wait(1)
