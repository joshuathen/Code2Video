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
        lecture_lines = ["The loop is: predict, measure, update.", "We repeat this to train the network.", "Eventually, the AI learns perfectly."]
        self.setup_layout("Integration & Conclusion", lecture_lines)
        
        # Elements
        steps = VGroup(
            Text("1. Predict", font_size=24),
            Text("2. Measure Cost", font_size=24),
            Text("3. Calculate Gradient", font_size=24),
            Text("4. Update Weights", font_size=24),
            Text("5. Repeat", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # FIX 1 (Issue 33, 40)
        self.place_in_area(steps, 'B1', 'C3', scale_factor=0.6)
        
        initial_state = Text("Initial State", color=BLUE, font_size=30)
        optimized_label = Text("Optimized", color=GREEN, font_size=30)
        
        # Asset integration
        network_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/network.svg")
        
        # FIX 2 (Issue 34, 40)
        self.place_at_grid(initial_state, 'D2', scale_factor=0.7)
        # FIX 3 (Issue 35, 40)
        self.place_at_grid(optimized_label, 'D3', scale_factor=0.8)
        
        self.place_at_grid(network_icon, 'E4', scale_factor=1.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(steps))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Indicate(steps))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Animate transition using asset
        self.play(
            ReplacementTransform(initial_state, optimized_label),
            FadeIn(network_icon)
        )
        
        # Flash loop
        flash_group = VGroup(*[steps[i] for i in range(5)])
        self.play(Flash(flash_group, color=YELLOW, line_length=0.2, num_lines=15))
        self.wait(2)
