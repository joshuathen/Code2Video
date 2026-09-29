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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Intuitive Hook: The Velocity-Position Analogy", [
            "Mars rover moves along a path.", 
            "Speedometer reading is the derivative.", 
            "Odometer distance is the integral."
        ])
        
        # Define objects
        # Using SVG Assets as requested
        rover = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rover.svg")
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        odometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/odometer.svg")
        
        position_line = NumberLine(x_range=[0, 6], length=4, include_numbers=True)
        position_label = Text("Position", color="#FFD700")
        velocity_label = Text("Rate of Change", color="#FF4500")

        # Placement
        self.place_in_area(position_line, 'C2', 'C6', scale_factor=0.9)
        self.place_at_grid(position_label, 'B1', scale_factor=0.8)
        
        self.place_at_grid(rover, 'C1')
        self.place_at_grid(velocity_label, 'D1', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(Create(position_line), Write(position_label), FadeIn(rover))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00CED1")
        self.play(FadeIn(speedometer), Write(velocity_label))
        self.play(rover.animate.shift(RIGHT * 3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        self.play(FadeIn(odometer))
        self.wait(1)
        self.play(FadeOut(rover), FadeOut(velocity_label), FadeOut(position_line), FadeOut(position_label), FadeOut(speedometer), FadeOut(odometer))
        self.wait(1)
