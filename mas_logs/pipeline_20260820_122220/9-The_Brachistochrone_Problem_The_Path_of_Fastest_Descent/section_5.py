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
        self.setup_layout("Conclusion and Application", [
            "This problem defined the calculus of variations.",
            "Its principles optimize modern engineering designs.",
            "Gravity-powered paths rely on this beautiful curve."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Cycloid shape for structural application
        cycloid = ParametricFunction(
            lambda t: np.array([t - np.sin(t), - (1 - np.cos(t)), 0]),
            t_range=[0, 2*PI]
        ).set_color("#FFFFFF")
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        # Apply positioning constraints
        self.place_in_area(cycloid, "A2", "B5", scale_factor=0.6)
        self.place_at_grid(bridge_icon, "A5", scale_factor=0.4)
        self.play(Create(cycloid), FadeIn(bridge_icon))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Real-world context: simplified roller coaster ramp
        ramp = Line(start=[-1, 0, 0], end=[1, -1, 0], color="#8A2BE2").scale(2)
        label = Text("Roller Coaster Drop", font_size=20, color="#8A2BE2")
        self.place_in_area(ramp, "D2", "E5", scale_factor=0.8)
        self.place_at_grid(label, "E3", scale_factor=0.8)
        self.play(Create(ramp), Write(label))
        self.lecture[1].set_color("#8A2BE2")

        # === Animation for Lecture Line 3 ===
        # Final summary
        summary = Text("Optimal Efficiency", font_size=32, color="#FFD700")
        pendulum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        self.place_at_grid(summary, "D3", scale_factor=0.9)
        self.place_at_grid(pendulum_icon, "D6", scale_factor=0.5)
        self.play(FadeIn(summary), FadeIn(pendulum_icon))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
