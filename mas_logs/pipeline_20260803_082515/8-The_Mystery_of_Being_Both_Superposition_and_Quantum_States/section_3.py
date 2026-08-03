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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining the Quantum State (|ψ⟩)", [
            "We call this quantum state vector 'Ket Psi'.",
            "It can point anywhere between our two outcome axes.",
            "Crucially, the length of this arrow is always one."
        ])

        psi_color = "#FF69B4"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(psi_color)
        
        # Create a unit circle and axes
        axes = Axes(
            x_range=[0, 1.2, 1],
            y_range=[0, 1.2, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        # Fix 40: Use area B2 to F5 for better utilization of vertical space
        self.place_in_area(axes, "B2", "F5")
        
        # The center of the axes is our origin
        origin = axes.c2p(0, 0)
        # Unit circle (arc for the first quadrant)
        unit_circle = Arc(radius=axes.get_x_unit_size(), start_angle=0, angle=PI/2, color=GRAY_A)
        unit_circle.move_to(origin, aligned_edge=DL)
        
        # Labels for the axes
        label_x = Text("Asleep", font_size=18).next_to(axes.x_axis.get_end(), DOWN)
        label_y = Text("Awake", font_size=18).next_to(axes.y_axis.get_end(), LEFT)
        
        psi_label = MathTex(r"|\psi\rangle", color=psi_color)
        # Fix 39: Place at B6, scale 1.0 to avoid overlap with axes area
        self.place_at_grid(psi_label, "B6", scale_factor=1.0)

        self.play(
            Create(axes),
            Create(unit_circle),
            Write(psi_label),
            Write(label_x),
            Write(label_y),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(psi_color)
        
        # Initial vector at 45 degrees
        angle_tracker = ValueTracker(45 * DEGREES)
        
        # Persistent vector object updated by ValueTracker
        vector_obj = Vector(axes.c2p(np.cos(45*DEGREES), np.sin(45*DEGREES)) - origin, color=psi_color).shift(origin)
        vector_obj.add_updater(lambda m: m.become(
            Vector(axes.c2p(np.cos(angle_tracker.get_value()), np.sin(angle_tracker.get_value())) - origin, color=psi_color).shift(origin)
        ))

        self.play(GrowArrow(vector_obj))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(psi_color)
        
        # Rotate smoothly to show it can point anywhere and maintain length
        self.play(
            angle_tracker.animate.set_value(15 * DEGREES),
            run_time=1.5,
            rate_func=smooth
        )
        self.play(
            angle_tracker.animate.set_value(75 * DEGREES),
            run_time=2,
            rate_func=smooth
        )
        self.play(
            angle_tracker.animate.set_value(45 * DEGREES),
            run_time=1.5,
            rate_func=smooth
        )
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
