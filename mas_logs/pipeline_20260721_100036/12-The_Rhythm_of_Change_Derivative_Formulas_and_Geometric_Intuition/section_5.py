from manim import *
import numpy as np

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
        # Setup the layout with the specific title and lecture lines
        self.setup_layout(
            "Application: The Cheetah’s Sprint",
            [
                "Derivatives describe real movement, like a running cheetah.",
                "Velocity is the derivative of position over time.",
                "Acceleration is the derivative of velocity, measuring speed changes."
            ]
        )
        
        # Define colors for consistency between text and visuals
        COLOR_POS = "#58ACFA" # Light Blue
        COLOR_VEL = "#58FA58" # Light Green
        COLOR_ACC = "#FA5858" # Light Red
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(COLOR_POS))
        
        # Cheetah representation (using a simple triangle)
        cheetah = Triangle(color=ORANGE, fill_opacity=1).rotate(-PI/2)
        self.place_at_grid(cheetah, "B1", scale_factor=0.3)
        
        # Position label s(t) that follows the cheetah
        pos_label = MathTex("s(t)", color=COLOR_POS)
        pos_label.add_updater(lambda m: m.next_to(cheetah, UP, buff=0.2))
        
        self.play(FadeIn(cheetah), FadeIn(pos_label))
        
        # Move cheetah from start to end of track (B1 to B6)
        dest_pos_end = self.grid["B6"]
        self.play(cheetah.animate.move_to(dest_pos_end), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_VEL)
        )
        
        # Velocity formula positioned in grid
        # Resolved Issue 37: Scale reduced to 0.7 to avoid crowding
        vel_formula = MathTex("v(t) = s'(t)", color=COLOR_VEL)
        self.place_at_grid(vel_formula, "C2", scale_factor=0.7)
        
        # Speedometer setup
        # Moved speedometer to E5 to make room for acc_formula at E2
        speedo_center = self.grid["E5"]
        speedo_arc = Arc(radius=0.7, start_angle=PI, angle=-PI, color=WHITE)
        speedo_arc.move_to(speedo_center)
        arc_center_point = speedo_arc.get_arc_center()
        
        # Needle and ValueTracker for synchronized speed display
        vt = ValueTracker(0)
        needle = Line(arc_center_point, arc_center_point + LEFT * 0.6, color=COLOR_VEL)
        
        def update_needle(m):
            angle = (1 - vt.get_value()) * PI
            new_end = arc_center_point + np.array([np.cos(angle), np.sin(angle), 0]) * 0.6
            m.set_points_as_corners([arc_center_point, new_end])
            
        needle.add_updater(update_needle)
        
        speedo_text = Text("Velocity", font_size=16, color=COLOR_VEL)
        speedo_text.next_to(speedo_arc, DOWN, buff=0.2)
        
        self.play(
            Create(speedo_arc),
            Create(needle),
            Write(vel_formula),
            FadeIn(speedo_text)
        )
        
        # Demonstrate variable velocity: reset and move again
        self.play(
            cheetah.animate.move_to(self.grid["B1"]), 
            vt.animate.set_value(0), 
            run_time=1
        )
        
        # Move with acceleration/deceleration rate_func
        self.play(
            cheetah.animate.move_to(dest_pos_end),
            vt.animate.set_value(1),
            run_time=4,
            rate_func=slow_into 
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Transition highlight to third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_ACC)
        )
        
        # Acceleration formula
        # Resolved Issue 36: Moved to E2 and scale reduced to 0.7 to avoid horizontal overlap
        acc_formula = MathTex("a(t) = v'(t)", color=COLOR_ACC)
        self.place_at_grid(acc_formula, "E2", scale_factor=0.7)
        
        # Acceleration arrow and label following the cheetah
        acc_arrow = Arrow(LEFT, RIGHT, color=COLOR_ACC, buff=0, stroke_width=4)
        acc_label = Text("Acceleration", font_size=16, color=COLOR_ACC)
        
        acc_arrow.add_updater(lambda m: m.next_to(cheetah, RIGHT, buff=0.1))
        acc_label.add_updater(lambda m: m.next_to(acc_arrow, UP, buff=0.1))
        
        self.play(Write(acc_formula), GrowArrow(acc_arrow), FadeIn(acc_label))
        
        # Final sprint showing high acceleration
        self.play(
            cheetah.animate.move_to(self.grid["B1"]),
            vt.animate.set_value(0),
            run_time=1
        )
        
        self.play(
            cheetah.animate.move_to(dest_pos_end),
            vt.animate.set_value(1),
            run_time=3,
            rate_func=rush_into # High acceleration profile
        )
        
        # Pulse animation for labels at top speed/acceleration
        self.play(
            speedo_text.animate.scale(1.3).set_color(WHITE),
            acc_label.animate.scale(1.3).set_color(WHITE),
            run_time=0.3
        )
        self.play(
            speedo_text.animate.scale(1/1.3).set_color(COLOR_VEL),
            acc_label.animate.scale(1/1.3).set_color(COLOR_ACC),
            run_time=0.3
        )
        self.play(
            speedo_text.animate.scale(1.3).set_color(WHITE),
            acc_label.animate.scale(1.3).set_color(WHITE),
            run_time=0.3
        )
        self.play(
            speedo_text.animate.scale(1/1.3).set_color(COLOR_VEL),
            acc_label.animate.scale(1/1.3).set_color(COLOR_ACC),
            run_time=0.3
        )

        self.wait(2)
