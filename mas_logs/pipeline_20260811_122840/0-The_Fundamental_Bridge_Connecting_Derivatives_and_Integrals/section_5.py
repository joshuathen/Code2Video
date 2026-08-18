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
        self.setup_layout("Summary & Application", [
            "The Integral recovers the original anti-derivative.", 
            "Solve complex physics using this inverse relationship.", 
            "Fuel rate leads to exact rocket altitude."
        ])
        
        # Assets
        rocket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg")
        fuel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fuel.svg")
        
        # === Animation for Lecture Line 1 ===
        # Display a rocket trajectory graph.
        axes = Axes(x_range=[0, 5], y_range=[0, 5], axis_config={"include_tip": False}).scale(0.4)
        curve = axes.plot(lambda x: 0.2 * x**2, color="#FFD700")
        self.place_in_area(axes, "C1", "F6", scale_factor=0.9)
        self.place_in_area(curve, "C1", "F6", scale_factor=0.9)
        
        self.place_at_grid(rocket, "A4", scale_factor=0.5)
        self.place_at_grid(fuel, "A5", scale_factor=0.5)
        
        self.play(Create(axes), Create(curve), FadeIn(rocket), FadeIn(fuel))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        # Overlay the velocity curve and area under it.
        area = axes.get_area(curve, x_range=[0, 4], color="#00CED1", opacity=0.5)
        self.place_in_area(area, "C1", "F6", scale_factor=0.9)
        self.play(FadeIn(area))
        self.lecture[1].set_color("#00CED1")

        # === Animation for Lecture Line 3 ===
        # Flash total distance reached by the rocket.
        label = Text("Altitude = \u222B v(t)dt", font_size=20, color="#FF4500")
        self.place_at_grid(label, "F4", scale_factor=1.0)
        self.play(Indicate(label))
        self.lecture[2].set_color("#FF4500")
        self.wait(2)
