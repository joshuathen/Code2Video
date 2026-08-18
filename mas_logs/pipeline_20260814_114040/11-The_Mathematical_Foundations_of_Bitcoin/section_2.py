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
        lecture_lines = [
            "Proof of Work validates network activity.",
            "Miners guess nonces for specific hash results.",
            "This trial and error requires immense effort."
        ]
        self.setup_layout("The Digital Puzzle: Proof of Work", lecture_lines)
        
        # Paths
        lock_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg"
        robot_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        
        # Assets
        lock = SVGMobject(lock_path).set_color("#FFD700")
        robots = VGroup(*[SVGMobject(robot_path).set_color("#00BFFF") for _ in range(3)])
        
        # === Animation for Lecture Line 1 ===
        # Use place_in_area as suggested by review
        self.place_in_area(lock, 'B4', 'B6', scale_factor=0.5)
        self.play(FadeIn(lock))
        self.play(self.lecture[0].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 2 ===
        # Position robots
        self.place_at_grid(robots, 'C3', scale_factor=0.3)
        self.play(FadeIn(robots))
        
        nonce_val = ValueTracker(0)
        nonce_text = Text("Nonce: 0", color="#00BFFF")
        nonce_text.add_updater(lambda m: m.become(Text(f"Nonce: {int(nonce_val.get_value())}", color="#00BFFF", font_size=24)))
        
        # Review suggested: line 63: self.place_at_grid(nonce_text, 'D5', scale_factor=0.4)
        self.place_at_grid(nonce_text, 'D5', scale_factor=0.4)
        self.add(nonce_text)
        
        self.play(nonce_val.animate.set_value(999), run_time=2)
        self.play(self.lecture[1].animate.set_color("#00BFFF"))

        # === Animation for Lecture Line 3 ===
        # Review suggested: line 70: self.place_at_grid(valid_hash, 'E5', scale_factor=0.4)
        valid_hash = Text("Hash: 0000a1b2...", color="#00FF00")
        self.place_at_grid(valid_hash, 'E5', scale_factor=0.4)
        
        lock.set_color("#00FF00")
        self.play(Write(valid_hash), lock.animate.set_color("#00FF00"))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(1)
