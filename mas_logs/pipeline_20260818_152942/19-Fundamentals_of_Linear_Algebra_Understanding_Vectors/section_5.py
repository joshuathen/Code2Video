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
        self.setup_layout("Application and Conclusion", [
            "Vectors are essential in computer graphics.",
            "They model physical forces in physics.",
            "Magnitude, direction, and combinations are key."
        ])
        
        # Load assets
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        monitor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg")
        
        # Elements
        thrust_vector = Arrow(ORIGIN, RIGHT * 1.5 + UP * 1.0, color=BLUE)
        wind_vector = Arrow(ORIGIN, RIGHT * 1.0 + DOWN * 1.0, color=YELLOW)
        thrust_label = Text("Thrust", font_size=20, color=BLUE)
        wind_label = Text("Wind", font_size=20, color=YELLOW)
        
        self.place_at_grid(computer_icon, "B2", scale_factor=0.5)
        self.place_at_grid(monitor_icon, "E2", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(computer_icon))
        self.place_at_grid(thrust_vector, "B2", scale_factor=0.6)
        self.place_at_grid(thrust_label, "B3", scale_factor=0.8)
        self.play(Create(thrust_vector), Write(thrust_label))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.play(FadeIn(monitor_icon))
        self.place_at_grid(wind_vector, "E2", scale_factor=0.6)
        self.place_at_grid(wind_label, "E3", scale_factor=0.8)
        self.play(Create(wind_vector), Write(wind_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        combined_vec = Arrow(ORIGIN, thrust_vector.get_vector() + wind_vector.get_vector(), color="#00FFFF")
        self.place_at_grid(combined_vec, "C4", scale_factor=0.6)
        self.play(Create(combined_vec))
        self.wait(1)
        self.play(FadeOut(VGroup(thrust_vector, wind_vector, combined_vec, thrust_label, wind_label, computer_icon, monitor_icon)))
