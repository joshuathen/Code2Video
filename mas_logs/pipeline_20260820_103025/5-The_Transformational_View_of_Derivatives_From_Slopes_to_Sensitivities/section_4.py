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
        self.setup_layout("The Dynamic Operator: Visualizing Change", [
            "View derivatives as dynamic, transforming operators.",
            "Position functions transform into velocity functions.",
            "Watch the graph morph in real-time."
        ])
        
        # Setup plot axes
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        self.place_in_area(axes, 'C2', 'E5', scale_factor=0.6)
        self.add(axes)

        # Assets
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg", color="#FFA500")
        label = Text("d/dx", font_size=36, color="#FFA500")
        gear_group = VGroup(gear, label).arrange(DOWN)
        
        curve = axes.plot(lambda x: x**2, color="#FF4500")
        result_curve = axes.plot(lambda x: 2*x, color="#7FFF00")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFA500"))
        self.place_at_grid(gear_group, 'B4', scale_factor=0.7)
        self.play(FadeIn(gear_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.play(Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#7FFF00"))
        
        # Ensure morphing happens within safe grid area
        graph_morph = VGroup(curve)
        
        self.play(
            Transform(curve, result_curve),
            Rotate(gear, angle=PI*2, about_point=gear.get_center())
        )
        self.wait(2)
