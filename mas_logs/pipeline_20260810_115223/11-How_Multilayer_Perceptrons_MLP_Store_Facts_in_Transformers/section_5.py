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
        self.setup_layout("Conclusion: The Distributed Nature of Facts", [
            "Facts are distributed across multiple neurons.", 
            "Robustness emerges from collective neuron activity.", 
            "Individual neurons don't store single facts."
        ])
        
        # Load Assets
        neuron_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg")
        
        # Create a group of neuron icons to represent distributed knowledge
        neurons = VGroup(*[neuron_svg.copy() for _ in range(9)])
        positions = ["B2", "B3", "B4", "C2", "C3", "C4", "D2", "D3", "D4"]
        for i, pos in enumerate(positions):
            self.place_at_grid(neurons[i], pos, scale_factor=0.3)
            neurons[i].set_color(BLUE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Fix: Using place_in_area as requested by Critic
        self.place_in_area(neurons, 'A4', 'F6', scale_factor=0.6)
        self.play(
            FadeIn(neurons),
            *[n.animate.set_color(ORANGE) for n in neurons],
            run_time=2
        )

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Grid title for line 2 as requested
        grid_title = Text("Neuron Activity", font_size=20, color=BLUE)
        self.place_at_grid(grid_title, 'A4', scale_factor=0.9)
        
        # Pulse neurons to show collective activity
        self.play(
            FadeIn(grid_title),
            *[n.animate.scale(1.5).set_color(WHITE) for n in neurons],
            *[n.animate.scale(1/1.5).set_color(ORANGE) for n in neurons],
            run_time=2
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Finalize with consolidated view
        grid_group = VGroup(grid_title, neurons)
        self.place_in_area(grid_group, 'B2', 'F5', scale_factor=0.75)
        
        self.play(
            FadeOut(grid_title),
            *[n.animate.set_opacity(0.3) for n in neurons],
            run_time=1
        )
        self.wait(1)
