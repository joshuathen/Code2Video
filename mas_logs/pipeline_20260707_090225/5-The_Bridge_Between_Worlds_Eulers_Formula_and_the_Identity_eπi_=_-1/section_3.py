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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialize Layout
        lecture_lines = [
            "Real powers of e represent continuous outward growth.",
            "But an imaginary power i changes the direction.",
            "Instead of growing larger, the value starts to turn.",
            "It is like a side-thruster creating a circular orbit.",
            "Constant speed, but always changing the movement angle."
        ]
        self.setup_layout("Growth vs. Rotation: The Role of 'e'", lecture_lines)

        # Assets
        thruster_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/thruster.svg"
        orbit_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/orbit.svg"

        # Coordinate System for the right side
        plane_center = self.grid["D3"]
        axes = Axes(
            x_range=[-2, 3, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "color": GREY_C},
            tips=False
        ).move_to(plane_center)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Dot representing growth
        growth_dot = Dot(color=WHITE, radius=0.08)
        growth_dot.move_to(axes.c2p(1, 0))
        
        # Thruster following the dot
        thruster = SVGMobject(thruster_asset).scale(0.15).set_color(WHITE)
        thruster.add_updater(lambda t: t.next_to(growth_dot, LEFT, buff=0.1))
        
        # Fading trail
        trail = TracedPath(growth_dot.get_center, stroke_color=WHITE, stroke_width=2, dissipating_time=0.5)
        
        # Label for growth (Issue 32: Positioning at A2)
        growth_label = Text("e^x", color=WHITE, font_size=30)
        self.place_at_grid(growth_label, 'A2', scale_factor=1.0)
        
        self.play(Create(axes), FadeIn(growth_dot), FadeIn(thruster), FadeIn(growth_label))
        self.add(trail)
        
        # Growth animation along the real axis
        growth_tracker = ValueTracker(1)
        growth_dot.add_updater(lambda d: d.move_to(axes.c2p(growth_tracker.get_value(), 0)))
        
        self.play(growth_tracker.animate.set_value(2.5), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF") # Cyan
        
        # Label for imaginary growth (Issue 30: Positioning at A4)
        rotation_label = Text("e^ix", color="#00FFFF", font_size=30)
        self.place_at_grid(rotation_label, 'A4', scale_factor=1.0)
        
        # Transition: return to 1 and change label
        self.play(
            FadeOut(growth_label),
            FadeIn(rotation_label),
            growth_tracker.animate.set_value(1),
            run_time=1
        )
        
        # Show "imaginary push" vector (perpendicular)
        push_arrow = Arrow(
            start=axes.c2p(1, 0),
            end=axes.c2p(1, 0.8),
            buff=0,
            color="#00FFFF",
            stroke_width=4
        )
        self.play(Create(push_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        # Tracking angle for rotation
        rot_tracker = ValueTracker(0)
        
        # Update dot and thruster for circular rotation
        growth_dot.clear_updaters()
        growth_dot.add_updater(lambda d: d.move_to(axes.c2p(np.cos(rot_tracker.get_value()), np.sin(rot_tracker.get_value()))))
        
        thruster.clear_updaters()
        def update_thruster(t):
            angle = rot_tracker.get_value()
            pos = axes.c2p(np.cos(angle), np.sin(angle))
            tangent = np.array([-np.sin(angle), np.cos(angle), 0])
            t.move_to(pos - tangent * 0.25)
            # Simple rotation based on angle
            t.set_angle(angle + PI/2)
        thruster.add_updater(update_thruster)
        
        # Update push arrow to follow tangent (representing side thrust)
        def update_arrow(a):
            angle = rot_tracker.get_value()
            p = axes.c2p(np.cos(angle), np.sin(angle))
            tangent_dir = np.array([-np.sin(angle), np.cos(angle), 0])
            a.put_start_and_end_on(p, p + tangent_dir * 0.7)
        push_arrow.add_updater(update_arrow)

        # Begin turning
        self.play(rot_tracker.animate.set_value(PI/2), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#C0C0C0") # Silver for orbit
        
        # Circular orbit path (Asset: orbit.svg)
        orbit = SVGMobject(orbit_asset).set_color("#C0C0C0").move_to(axes.c2p(0,0))
        # Scale to match unit circle on axes
        unit_radius = axes.c2p(1,0)[0] - axes.c2p(0,0)[0]
        orbit.scale_to_fit_height(unit_radius * 2)
        
        # Label "Imaginary Push" (Issue 31: Area placement B4-B5)
        push_text = Text("Imaginary Push", color="#00FFFF", font_size=24)
        self.place_in_area(push_text, 'B4', 'B5', scale_factor=0.7)
        
        self.play(FadeIn(orbit), Write(push_text))
        self.play(rot_tracker.animate.set_value(PI), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00") # Green for angle
        
        # Radius line showing constant distance
        radius_line = Line(axes.c2p(0,0), axes.c2p(1,0), color=WHITE, stroke_width=2)
        radius_line.add_updater(lambda r: r.set_points_as_corners([axes.c2p(0,0), growth_dot.get_center()]))
        
        # Angle indicator (Arc and theta label)
        angle_arc = Arc(radius=0.4, start_angle=0, angle=0.01, arc_center=axes.c2p(0,0), color="#00FF00")
        angle_arc.add_updater(lambda a: a.become(Arc(
            radius=0.4, 
            start_angle=0, 
            angle=max(0.01, rot_tracker.get_value() % (2*PI)), 
            arc_center=axes.c2p(0,0), 
            color="#00FF00"
        )))
        
        theta_label = Text("θ", color="#00FF00", font_size=24)
        def update_theta(l):
            angle = (rot_tracker.get_value() % (2*PI)) / 2
            l.move_to(axes.c2p(0.6 * np.cos(angle), 0.6 * np.sin(angle)))
        theta_label.add_updater(update_theta)
        
        self.add(radius_line, angle_arc, theta_label)
        self.play(rot_tracker.animate.set_value(2*PI), run_time=3, rate_func=linear)
        self.wait(2)
