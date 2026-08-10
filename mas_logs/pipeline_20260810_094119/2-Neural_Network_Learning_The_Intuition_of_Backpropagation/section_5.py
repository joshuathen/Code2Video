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
        self.setup_layout("Conclusion: The Iterative Loop", [
            "Predict, check, adjust, and repeat the cycle.", 
            "Thousands of iterations build an expert network.", 
            "Learning is a repetitive practice of refinement."
        ])
        
        # Assets
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")
        network = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/network.svg")
        
        # Visuals
        loop = Circle(radius=1.2, color=WHITE)
        forward = Text("Forward", font_size=20, color=WHITE)
        loss = Text("Loss", font_size=20, color=WHITE)
        backward = Text("Backward", font_size=20, color=WHITE)
        weights = Text("Weights Update", font_size=20, color=WHITE)
        
        # Positioning Fixes (Issues 33, 34, 35, 40)
        # Using grid to avoid overlap and improve hierarchy
        self.place_at_grid(loop, 'D3', scale_factor=0.7)
        self.place_at_grid(brain, 'D3', scale_factor=0.5)
        
        # Position labels around the loop
        forward.next_to(loop, UP, buff=0.1)
        loss.next_to(loop, RIGHT, buff=0.1)
        backward.next_to(loop, DOWN, buff=0.1)
        self.place_at_grid(weights, 'D4', scale_factor=0.6)
        
        practice_text = Text("Practice makes perfect!", font_size=28, color="#00FF00")
        self.place_at_grid(practice_text, 'F3', scale_factor=0.8)

        # Animation logic
        # === Animation for Lecture Line 1 ===
        self.play(Create(loop), FadeIn(brain), FadeIn(forward), FadeIn(loss), FadeIn(backward), FadeIn(weights))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Repeat loop with network asset
        self.play(FadeIn(network.scale(0.5).move_to(loop)))
        self.play(Rotate(loop, angle=2*PI, run_time=2))
        self.play(FadeOut(network))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Write(practice_text))
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
