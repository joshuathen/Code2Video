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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Future: Feature Maps in AI", [
            "Deep learning learns these kernel values automatically.",
            "AI uses these features to recognize objects.",
            "Computers now see the world like humans."
        ])
        self.lecture.set_opacity(0)
        
        # Asset definitions
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color=WHITE)
        eye = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg", color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.lecture[0].set_color("#88CCFF")
        
        # Visualizing feature maps
        feature_maps = VGroup(*[Square(side_length=0.6, color=WHITE, fill_opacity=0.3) for _ in range(4)])
        feature_maps.arrange(RIGHT, buff=0.2)
        self.place_at_grid(robot, "A4", scale_factor=0.6)
        self.place_in_area(feature_maps, "A2", "B5", scale_factor=0.75)
        self.play(FadeIn(robot), Create(feature_maps))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.lecture[1].set_color("#FFFF88")
        
        # Stacking maps into a neural network
        network_stack = VGroup(*[Square(side_length=0.6, color=BLUE, fill_opacity=0.5) for _ in range(3)])
        network_stack.arrange(DOWN, buff=-0.3)
        self.place_in_area(network_stack, "C2", "D3", scale_factor=0.85)
        self.play(FadeIn(network_stack))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        self.lecture[2].set_color("#88FF88")
        
        # Object classification result
        result_label = Text("Dog", font_size=36, color=GREEN).add_background_rectangle()
        self.place_at_grid(result_label, "E4", scale_factor=0.9)
        self.place_at_grid(eye, "E3", scale_factor=0.5)
        self.play(Write(result_label), FadeIn(eye), run_time=1.5)
        self.wait(2)
