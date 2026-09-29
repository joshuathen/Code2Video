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
        self.setup_layout("Defining the Binomial Distribution", [
            "BINS defines our criteria.",
            "Binary outcomes, independent trials.",
            "Fixed trials, constant probability."
        ])
        
        # === Animation for Lecture Line 1 ===
        # BINS defines our criteria.
        self.lecture[0].set_color("#00FFFF")
        
        # Visualizing trial blocks with Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg
        trials = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg") for _ in range(5)])
        trials.arrange(RIGHT, buff=0.2)
        # Using B4 as requested to fix clutter
        self.place_at_grid(trials, "B4", scale_factor=0.8)
        self.play(Create(trials))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Binary outcomes, independent trials.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF00FF")
        
        # Highlighting successes (k)
        successes = VGroup(trials[1], trials[3])
        self.play(successes.animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fixed trials, constant probability.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        # Show labels n and k
        label_n = Text("n=5", font_size=24, color=WHITE)
        label_k = Text("k=2", font_size=24, color="#FF00FF")
        
        # Using C4 and C5 as requested to fix scattered labels
        self.place_at_grid(label_n, "C4", scale_factor=1.0)
        self.place_at_grid(label_k, "C5", scale_factor=1.0)
        
        self.play(Write(label_n), Write(label_k))
        self.wait(2)
