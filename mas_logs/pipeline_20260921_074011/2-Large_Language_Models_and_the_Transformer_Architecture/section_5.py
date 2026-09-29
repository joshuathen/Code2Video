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
        self.setup_layout("Practical Application: Next-Token Prediction", [
            "Models predict the next likely token.",
            "They function as advanced probability engines.",
            "Visualizing the selection process."
        ])
        
        # Elements
        prompt = Text("The sky is...", font_size=36)
        token_box = Rectangle(width=1.5, height=0.8, color=WHITE)
        result = Text("blue", font_size=36, color=GREEN)
        
        # Assets
        slot_machine_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slotmachine.svg")
        slot_machine_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slotmachine.svg")
        
        # Position initial state
        self.place_at_grid(prompt, 'B2', scale_factor=0.7)
        self.place_at_grid(token_box, 'B5', scale_factor=0.7)
        self.place_at_grid(slot_machine_1, 'B4', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Write(prompt), Create(token_box), FadeIn(slot_machine_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Probability representation
        dist_box = Rectangle(width=2.5, height=2.0, color=YELLOW).set_fill(YELLOW, opacity=0.1)
        self.place_in_area(dist_box, 'C4', 'E6', scale_factor=0.8)
        self.play(FadeIn(dist_box))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.place_at_grid(result, 'B5', scale_factor=0.7)
        self.place_at_grid(slot_machine_2, 'B6', scale_factor=0.5)
        self.play(Write(result), FadeIn(slot_machine_2))
        self.wait(2)
