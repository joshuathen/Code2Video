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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: The MLP as a Knowledge Base", [
            "MLPs are vast distributed databases.",
            "Training saves facts into weights.",
            "We can surgically update these weights."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Visualize MLP as a grid of weights
        weights = VGroup(*[Dot(radius=0.1, color=BLUE) for _ in range(25)])
        weights.arrange_in_grid(5, 5, buff=0.2)
        # Applying fix for Issue 33 & 35 (VideoCritic layout improvements)
        self.place_in_area(weights, "B3", "E5", scale_factor=0.85)
        self.play(FadeIn(weights))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Show facts being saved (highlighting a few dots)
        facts = VGroup(weights[7], weights[12], weights[17])
        self.play(facts.animate.set_color(YELLOW))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Surgical edit (change one weight)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/scalpel.svg]
        scalpel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scalpel.svg")
        self.place_at_grid(scalpel, "C4", scale_factor=0.8)
        
        self.play(FadeIn(scalpel), weights[12].animate.set_color(RED))
        self.play(FadeOut(scalpel))
        self.lecture[2].set_color(RED)
        
        # Applying fix for Issue 34 (VideoCritic layout improvements)
        final_label = Text("MLP: A Distributed Knowledge Base", font_size=24, color=WHITE)
        self.place_at_grid(final_label, "F4", scale_factor=0.75)
        self.play(Write(final_label))
        self.wait(2)
