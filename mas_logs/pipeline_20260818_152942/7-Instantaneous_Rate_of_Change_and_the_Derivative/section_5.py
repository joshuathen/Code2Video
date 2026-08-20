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
        lecture_lines = [
            "Derivatives reveal local function behavior.",
            "Used for velocity, growth, and cost.",
            "Essential for understanding real-world systems.",
            "Calculate exact speeds for moving objects.",
            "Apply math to dynamic, changing phenomena."
        ]
        self.setup_layout("Application: Real-world Significance", lecture_lines)
        
        # Elements
        graph = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"color": BLUE}).scale(0.5)
        func = graph.plot(lambda x: 0.5 * x**2, color="#00FFFF")
        vel_label = Text("Velocity", font_size=20, color="#00FFFF")
        
        # Asset Loading
        vehicle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vehicle.svg")
        machine_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg")
        
        accel_arrow = Arrow(start=ORIGIN, end=UP*0.5, color=RED)
        accel_label = Text("Acceleration", font_size=20, color=RED)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.place_at_grid(graph, 'B2')
        self.place_at_grid(func, 'B2')
        self.place_at_grid(vehicle_icon, 'C2', scale_factor=0.3)
        self.place_in_area(vel_label, 'A4', 'A5', scale_factor=0.7)
        self.play(Create(graph), Create(func), FadeIn(vehicle_icon), Write(vel_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        self.place_at_grid(accel_arrow, 'C3', scale_factor=0.7)
        self.place_at_grid(accel_label, 'D3', scale_factor=0.7)
        self.place_at_grid(machine_icon, 'D4', scale_factor=0.3)
        self.play(Create(accel_arrow), Write(accel_label), FadeIn(machine_icon))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.wait(2)
