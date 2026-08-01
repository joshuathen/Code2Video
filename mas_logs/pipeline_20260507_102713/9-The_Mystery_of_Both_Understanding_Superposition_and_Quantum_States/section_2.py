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

class Section2Scene(TeachingScene):
    def construct(self):
        # Define the lecture script
        lecture_lines = [
            'We represent the state zero as a vertical vector.', 
            'The state one is represented by a horizontal vector.', 
            'A superposition vector points somewhere in between.', 
            'Projections show the contribution from each basis state.', 
            "The vector shifts as the system's state changes."
        ]
        
        self.setup_layout("Prerequisite: The Vector State Representation", lecture_lines)

        # Colors
        COLOR_0 = "#00FF00" # Green
        COLOR_1 = "#0000FF" # Blue
        COLOR_PSI = "#FFFF00" # Yellow
        COLOR_PROJ = "#888888" # Gray
        
        # Assets
        VECTOR_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/vector.svg"

        # Origin point for our coordinate system
        origin = self.grid["E2"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_0))

        # Vertical green arrow for |0> (Y-axis)
        arrow_0 = Arrow(start=origin, end=self.grid["B2"], color=COLOR_0, buff=0)
        label_0 = Text("|0⟩", color=COLOR_0, font_size=32)
        # Fix: Move label_0 to A2 to avoid overlap with arrowhead at B2
        self.place_at_grid(label_0, "A2", scale_factor=0.8)

        self.play(
            GrowArrow(arrow_0),
            Write(label_0),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_1))

        # Horizontal blue arrow for |1> (X-axis)
        arrow_1 = Arrow(start=origin, end=self.grid["E5"], color=COLOR_1, buff=0)
        label_1 = Text("|1⟩", color=COLOR_1, font_size=32)
        # Fix: Move label_1 to E6 to avoid overlap with arrowhead at E5
        self.place_at_grid(label_1, "E6", scale_factor=0.8)

        self.play(
            GrowArrow(arrow_1),
            Write(label_1),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_PSI))

        # State vector |psi> using Asset
        angle_tracker = ValueTracker(45 * DEGREES)
        psi_len = 3.0
        
        # Vector asset setup
        psi_svg = SVGMobject(VECTOR_ASSET)
        psi_svg.set_color(COLOR_PSI)
        psi_svg.set_height(psi_len)
        # Align bottom of SVG to origin
        psi_svg.move_to(origin, aligned_edge=DOWN)
        
        # Initial rotation to 45 degrees
        psi_svg.rotate(45 * DEGREES - 90 * DEGREES, about_point=origin) # Assuming SVG points UP (90deg)

        label_psi = Text("|ψ⟩", color=COLOR_PSI, font_size=32)
        
        def update_label_psi(m):
            # Calculate tip based on tracker
            ang = angle_tracker.get_value()
            tip = origin + np.array([np.cos(ang), np.sin(ang), 0]) * psi_len
            m.move_to(tip + 0.4 * (tip - origin) / psi_len)

        label_psi.add_updater(update_label_psi)

        # We'll use a functional way to update the SVG orientation
        def update_psi_svg(m):
            ang = angle_tracker.get_value()
            # Reset orientation and then apply new rotation
            m.set_height(psi_len)
            m.move_to(origin, aligned_edge=DOWN)
            # Standard SVG is UP (90 deg), so we rotate relative to that
            m.rotate(ang - 90 * DEGREES, about_point=origin)

        psi_svg.add_updater(update_psi_svg)

        self.play(
            FadeIn(psi_svg, scale=0.1, shift=origin),
            Write(label_psi),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_PROJ))

        # Projection lines
        def get_proj_x():
            ang = angle_tracker.get_value()
            tip = origin + np.array([np.cos(ang), np.sin(ang), 0]) * psi_len
            return DashedLine(
                start=tip,
                end=np.array([tip[0], origin[1], 0]),
                color=COLOR_PROJ,
                stroke_width=2
            )

        def get_proj_y():
            ang = angle_tracker.get_value()
            tip = origin + np.array([np.cos(ang), np.sin(ang), 0]) * psi_len
            return DashedLine(
                start=tip,
                end=np.array([origin[0], tip[1], 0]),
                color=COLOR_PROJ,
                stroke_width=2
            )

        proj_x = always_redraw(get_proj_x)
        proj_y = always_redraw(get_proj_y)

        self.play(
            Create(proj_x),
            Create(proj_y),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_PSI))

        # Rotate |psi> towards |0> (90 degrees) then back towards |1> (15 degrees)
        self.play(
            angle_tracker.animate.set_value(80 * DEGREES),
            run_time=2,
            rate_func=slow_into
        )
        self.play(
            angle_tracker.animate.set_value(10 * DEGREES),
            run_time=2,
            rate_func=slow_into
        )
        self.play(
            angle_tracker.animate.set_value(45 * DEGREES),
            run_time=1.5
        )
        self.wait(2)
