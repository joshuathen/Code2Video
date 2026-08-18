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
        self.setup_layout("Summary: Instantaneous Change", [
            "Derivative measures instantaneous behavior everywhere.",
            "Turns global curve data into local slope.",
            "Position to velocity: derivative is the key."
        ])
        
        # Setup visuals
        axes = Axes(x_range=[0, 4], y_range=[0, 4], axis_config={"include_numbers": False}).scale(0.4)
        curve = axes.plot(lambda x: 0.2*x**3, color="#8A2BE2")
        
        # 2. Local slope at a point
        point = Dot(color="#FFD700")
        point.move_to(axes.c2p(2, 0.2*2**3))
        slope_line = Line(start=LEFT, end=RIGHT, color="#00FFFF").scale(0.3).rotate(np.arctan(0.2*3*2**2))
        slope_line.next_to(point, UP, buff=0.1) # B011: tether labels/objects
        
        # 3. Label symbols
        global_label = Text("Global", color="#8A2BE2", font_size=20)
        local_label = Text("Local", color="#FFD700", font_size=20)
        
        # Positioning based on critic feedback
        self.place_in_area(axes, 'C4', 'E6', scale_factor=1.0)
        self.place_in_area(curve, 'C4', 'E6', scale_factor=1.0)
        self.place_at_grid(point, 'C5', scale_factor=0.8)
        self.place_at_grid(slope_line, 'C4', scale_factor=0.9)
        self.place_at_grid(global_label, 'B3', scale_factor=0.7) # B020
        self.place_at_grid(local_label, 'D4', scale_factor=0.7) # B020

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color("#8A2BE2")
        self.play(Create(curve), FadeIn(global_label))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[1].set_color("#FFD700")
        self.play(Create(point), Create(slope_line), FadeIn(local_label))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_color("#00FFFF")
        self.play(Indicate(slope_line))
        self.wait(2)
